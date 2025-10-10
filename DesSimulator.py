import simpy
from typing import Dict, List
from collections import defaultdict
from BasicDefinitions import Task, DPU, Link, Placement, Resource
import math

class GlobalRouter: 
    '''
    可以调用get_path(u, v)，得到从dpu_x的u元件到dpu_y的v元件的最短路径
    '''
    def __init__(self, network: Dict[str, DPU], links: Dict[str, Link]):
        self.resources: List[str] = [res_id for dpu in network.values() for res_id in dpu.resources]
        self.res_map = {name: i for i, name in enumerate(self.resources)}
        # 对于(res: str, res_id: int), self.resources[res_id] = res, self.res_map[res] = res_id
        self.num_res = len(self.resources)
        # 初始化
        self.dist = [[float('inf')] * self.num_res for _ in range(self.num_res)]
        self.next_hop = [[-1] * self.num_res for _ in range(self.num_res)]
        for i in range(self.num_res): 
            self.dist[i][i] = 0
            self.next_hop[i][i] = i
        # 连接单个dpu内部的边
        for dpu in network.values():
            for u_name, v_name in dpu.noc:
                if u_name in self.res_map and v_name in self.res_map:
                    u, v = self.res_map[u_name], self.res_map[v_name]
                    self.dist[u][v] = 1
                    self.dist[v][u] = 1
                    self.next_hop[u][v] = v
                    self.next_hop[v][u] = u
        # 连接dpu间的边
        for link in links.values():
            u_nic, v_nic = f"{link.source_dpu}_nic", f"{link.dest_dpu}_nic"
            if u_nic in self.res_map and v_nic in self.res_map:
                u, v = self.res_map[u_nic], self.res_map[v_nic]
                self.dist[u][v] = 1
                self.dist[v][u] = 1
                self.next_hop[u][v] = v
                self.next_hop[v][u] = u
        self._floyd_warshall()

    def _floyd_warshall(self):
        for k in range(self.num_res):
            for i in range(self.num_res):
                for j in range(self.num_res):
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]
                        self.next_hop[i][j] = self.next_hop[i][k]
    
    def get_path(self, source_res_id: str, dest_res_id: str) -> List[str]:
        if source_res_id == dest_res_id: 
            return [source_res_id]
        if source_res_id not in self.res_map or dest_res_id not in self.res_map: 
            return []
        u, v = self.res_map[source_res_id], self.res_map[dest_res_id]
        if self.next_hop[u][v] == -1: 
            return []
        path_indices = [u]
        while u != v: 
            u = self.next_hop[u][v]
            path_indices.append(u)
        return [self.resources[i] for i in path_indices]

class SharedBus:
    def __init__(self, env, name, bandwidth_MBps):
        self.env = env
        self.name = name
        self.bandwidth_MB_us = bandwidth_MBps / 1e6 #所有的时间单位都以us计算
        self.active_transfers = []
        self.process = env.process(self.run())
        self.wakeup_event = env.event()
        self.time_of_last_update = 0.0
        self.num_active_at_last_update = 0

    def run(self):
        self.time_of_last_update = self.env.now
        while True:
            time_passed = self.env.now - self.time_of_last_update
            if time_passed > 1e-9 and self.num_active_at_last_update > 0:
                effective_bw = self.bandwidth_MB_us / self.num_active_at_last_update
                data_transferred = effective_bw * time_passed
                for t in self.active_transfers: t['remaining'] -= data_transferred
            new_active_list = []
            for t in self.active_transfers:
                if t['remaining'] > 1e-9: 
                    new_active_list.append(t)
                else:
                    t['remaining'] = 0
                    if not t['event'].triggered: 
                        t['event'].succeed()
            self.active_transfers = new_active_list
            self.time_of_last_update = self.env.now
            self.num_active_at_last_update = len(self.active_transfers)
            try:
                if not self.active_transfers:
                    yield self.wakeup_event # 进程挂起，wakeup_event.succeed()唤醒
                    self.wakeup_event = self.env.event()
                    continue
                effective_bw = self.bandwidth_MB_us / len(self.active_transfers)
                if effective_bw <= 1e-12: 
                    yield self.env.timeout(1e9) #如果传输数量过多，设置一个很长的超时，防止模拟出问题
                    continue
                times_to_finish = [t['remaining'] / effective_bw for t in self.active_transfers]
                time_to_next_completion = max(0, min(times_to_finish))
                yield self.env.timeout(time_to_next_completion)
            except simpy.Interrupt: pass
            
    def transfer(self, data_size_mb):
        if data_size_mb <= 1e-9: 
            return self.env.timeout(0)
        completion_event = self.env.event()
        was_idle = (len(self.active_transfers) == 0)
        self.active_transfers.append({'remaining': data_size_mb, 'event': completion_event})
        if not self.process.is_alive: 
            self.process = self.env.process(self.run())
        if was_idle: #空
            if not self.wakeup_event.triggered: 
                self.wakeup_event.succeed()
        else: #非空
            self.process.interrupt()
        return completion_event

class Simulator:
    def __init__(self, dag: Dict[str, Task], placement: Placement, network: Dict[str, DPU], links: Dict[str, Link], router: GlobalRouter): 
        self.dag = dag
        self.placement = placement
        self.network = network # network: dpus
        self.links = links
        self.router = router
        self._res_to_dpu_map = {}
        for dpu in self.network.values():
            for res_id in dpu.resources: 
                self._res_to_dpu_map[res_id] = dpu.id
            
    def _get_resource_dpu_id(self, resource_id: str) -> str:
        return self._res_to_dpu_map.get(resource_id)
    
    def start_simulation(self):
        '''
        为每个dpu元件建立simpy的资源
        为noc和dpu间的links建立SharedBus
        这些都存储在self.resources中
        '''
        self.env = simpy.Environment()
        self.resources = {}
        NOC_BANDWIDTH_MBPS = 100_000
        for dpu_id, dpu in self.network.items():
            self.resources[f"{dpu_id}_noc_bus"] = SharedBus(self.env, f"{dpu_id}_noc_bus", NOC_BANDWIDTH_MBPS)
            for res in dpu.resources.values():
                if res.type == 'compute': 
                    core_pool = simpy.Store(self.env, capacity=res.capacity)
                    for i in range(res.capacity):
                        core_pool.put(f"core_{i}")
                    self.resources[res.id] = core_pool
                elif res.bandwidth_mbps > 0: 
                    self.resources[res.id] = SharedBus(self.env, res.id, res.bandwidth_mbps)
        for link in self.links.values():
            bandwidth_MBps = link.bandwidth_gbps * 125
            link_key = tuple(sorted((f"{link.source_dpu}_nic", f"{link.dest_dpu}_nic"))) # 保证link_key唯一
            self.resources[link_key] = SharedBus(self.env, link.id, bandwidth_MBps)
            
        self.task_du_done_events = defaultdict(list)
        for task_id, task in self.dag.items():
            num_dus = task.du_num if task.du_num > 0 else 1
            self.task_du_done_events[task_id] = [self.env.event() for _ in range(num_dus)]

        self.du_arrival_stores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None))) # ???
        self.env.run(self.run()) # 开始模拟
        return self.env.now
    
    def run(self):
        for task_id in self.dag: 
            self.env.process(self._task_process_router(task_id))
        final_tasks = [tid for tid, task in self.dag.items() if not task.children]
        if not final_tasks: 
            return self.env.timeout(0)
        final_events = [ev for tid in final_tasks for ev in self.task_du_done_events[tid]]
        return simpy.AllOf(self.env, final_events)
    
    def _task_process_router(self, task_id: str):
        task = self.dag[task_id]
        if task.type == 'compute': 
            yield self.env.process(self._master_process(task_id, self._compute_worker))
        elif task.type == 'data': 
            yield self.env.process(self._master_process(task_id, self._data_worker))

    def _master_process(self, task_id: str, worker_func):
        task = self.dag[task_id]
        child_du_num = task.du_num if task.du_num > 0 else 1
        pipelined_parents, non_pipelined_parents = [], []
        for parent_id in task.parents:
            parent_task = self.dag[parent_id]
            parent_du_num = parent_task.du_num if parent_task.du_num > 0 else 1
            (pipelined_parents if parent_du_num == child_du_num else non_pipelined_parents).append(parent_id)
        for parent_id in task.parents:
            parent_task = self.dag[parent_id]
            parent_du_num = parent_task.du_num if parent_task.du_num > 0 else 1
            for i in range(parent_du_num): # 传输从parent_id到当前节点的所有du
                self.env.process(self._data_transfer_worker_du(parent_id, task_id, i))
        for i in range(child_du_num):
            self.env.process(worker_func(task_id, i, pipelined_parents, non_pipelined_parents))
        yield simpy.AllOf(self.env, self.task_du_done_events[task_id])

    def _compute_worker(self, task_id: str, du_index: int, pipelined_parents: List[str], non_pipelined_parents: List[str]):
        task = self.dag[task_id]
        yield self.env.process(self._wait_for_parent_data(task_id, du_index, pipelined_parents, non_pipelined_parents))
        compute_res_id = self.placement[task_id]
        compute_resource_pool = self.resources[compute_res_id]
        # get workload
        if hasattr(task, 'workload'):
            du_compute_time = task.workload / (task.du_num or 1)
        else:
            workload = {'linear': 10, 'slice': 2, 'rope': 15, 'view': 1, 'einsum': 25, 'add': 2, 'softmax': 8}
            base_workload = workload.get(task.compute_type, 5)
            du_compute_time = base_workload / (task.du_num or 1)
        
        core_token = yield compute_resource_pool.get()  # 请求一个核心
        try:
            # print(f"[{self.env.now:8.2f}] [ComputeWorker] Task '{task.id}' DU-{du_index}: Got core, starting compute on '{compute_res_id}'.")
            if du_compute_time > 0:
                yield self.env.timeout(du_compute_time)
        finally:
            # print(f"[{self.env.now:8.2f}] [ComputeWorker] Task '{task.id}' DU-{du_index}: Computation done, returning core.")
            yield compute_resource_pool.put(core_token) # 归还核心

        self.task_du_done_events[task_id][du_index].succeed()

    def _data_worker(self, task_id: str, du_index: int, pipelined_parents: List[str], non_pipelined_parents: List[str]):
        task = self.dag[task_id]
        if not task.parents:
            du_interval = getattr(task, 'du_interval_us', 0)
            if du_interval > 0 and du_index > 0:
                yield self.env.timeout(du_index * du_interval)
        
        yield self.env.process(self._wait_for_parent_data(task_id, du_index, pipelined_parents, non_pipelined_parents))

        storage_res_id = self.placement[task_id]; storage_bus = self.resources.get(storage_res_id)
        du_size = task.du_size if task.du_size > 0 else (task.data_size / (task.du_num or 1))
        if du_size > 0 and storage_bus:
            # print(f"[{self.env.now:8.2f}] [DataWorker] Task '{task.id}' DU-{du_index}: Starting storage op ({du_size:.2f} MB) on '{storage_res_id}'.")
            yield storage_bus.transfer(du_size)
        self.task_du_done_events[task_id][du_index].succeed()
        # print(f"[{self.env.now:8.2f}] [DataWorker] Task '{task.id}' DU-{du_index}: Finished storage op.")

    def _wait_for_parent_data(self, task_id: str, du_index: int, pipelined_parents: List[str], non_pipelined_parents: List[str]):
        wait_for_events = []
        for parent_id in pipelined_parents:
            if self.du_arrival_stores[parent_id][task_id][du_index] is None:
                self.du_arrival_stores[parent_id][task_id][du_index] = simpy.Store(self.env, capacity=1)
            wait_for_events.append(self.du_arrival_stores[parent_id][task_id][du_index].get())
        for parent_id in non_pipelined_parents:
            parent_task = self.dag[parent_id]
            parent_du_num = parent_task.du_num if parent_task.du_num > 0 else 1
            for i in range(parent_du_num):
                if self.du_arrival_stores[parent_id][task_id][i] is None:
                    self.du_arrival_stores[parent_id][task_id][i] = simpy.Store(self.env, capacity=1)
                wait_for_events.append(self.du_arrival_stores[parent_id][task_id][i].get())
        if wait_for_events: 
            yield simpy.AllOf(self.env, wait_for_events)

    def _data_transfer_worker_du(self, source_id: str, dest_id: str, du_index: int):
        '''
        模拟一个du从source_id到dest_id经过的传输过程
        '''
        source_task = self.dag[source_id]
        du_size = source_task.du_size if source_task.du_size > 0 else (source_task.data_size / (source_task.du_num or 1))

        if du_size <= 1e-9:
            if self.du_arrival_stores[source_id][dest_id][du_index] is None:
                self.du_arrival_stores[source_id][dest_id][du_index] = simpy.Store(self.env, capacity=1)
            yield self.env.timeout(0)
            yield self.du_arrival_stores[source_id][dest_id][du_index].put(True)
            return
        
        yield self.task_du_done_events[source_id][du_index]
        source_res_id = self.placement[source_id]
        dest_res_id = self.placement[dest_id]
        comm_path = self.router.get_path(source_res_id, dest_res_id)
        for j in range(len(comm_path) - 1):
            u_res, v_res = comm_path[j], comm_path[j+1]
            u_dpu = self._get_resource_dpu_id(u_res); v_dpu = self._get_resource_dpu_id(v_res)
            bus_resource = None
            if u_dpu == v_dpu: 
                bus_resource = self.resources.get(f"{u_dpu}_noc_bus")
            else:
                link_key = tuple(sorted((f"{u_dpu}_nic", f"{v_dpu}_nic")))
                bus_resource = self.resources.get(link_key)
            if bus_resource: 
                yield bus_resource.transfer(du_size)
            elif u_dpu != v_dpu: 
                print(f"Warning: No bus resource found for transfer from {u_res} to {v_res}")
        if self.du_arrival_stores[source_id][dest_id][du_index] is None:
            self.du_arrival_stores[source_id][dest_id][du_index] = simpy.Store(self.env, capacity=1)
        yield self.du_arrival_stores[source_id][dest_id][du_index].put(True)


if __name__ == '__main__':
    def run_simulator_verification_test():
        print("\n" + "="*50)
        print("Running Standalone Verification Test for DesSimulator (Corrected Interval Logic)")
        print("="*50)

        # 【已修改】测试用例: A (source) -> B (intermediate data) -> C (compute)
        # A是源头，应用du_interval。B是中间节点，不应用。
        dag: Dict[str, Task] = {
            "T_A": Task(id="T_A", name="Source", type='data', data_size=50.0, du_size=25.0, du_num=2, children=["T_B"]),
            "T_B": Task(id="T_B", name="IntermediateStore", type='data', data_size=50.0, du_size=25.0, du_num=2, parents=["T_A"], children=["T_C"]),
            "T_C": Task(id="T_C", name="Process", type='compute', data_size=0, du_size=0, du_num=2, compute_type='linear', parents=["T_B"])
        }
        # 只有源头节点A有du_interval
        setattr(dag["T_A"], 'du_interval_us', 10.0) 
        # T_B不设置du_interval
        setattr(dag["T_C"], 'workload', 20.0) # 每个DU计算10us

        dram_bw = 20_000; noc_bw = 100_000
        network: Dict[str, DPU] = {
            "dpu1": DPU("dpu1", resources={"dpu1_dram": Resource("dpu1_dram", "dram", "storage", bandwidth_mbps=dram_bw), "dpu1_arm": Resource("dpu1_arm", "arm", "compute", capacity=1)}, noc=[("dpu1_dram", "dpu1_arm")])
        }
        links: Dict[str, Link] = {}
        # 所有任务都在同一个DPU内，简化传输，聚焦于du_interval逻辑
        placement: Placement = {"T_A": "dpu1_dram", "T_B": "dpu1_dram", "T_C": "dpu1_arm"}

        # --- Theoretical Calculation ---
        # T_A (Source)
        du_size_A = 25.0; interval_A = 10.0; time_dram_du_A = (du_size_A / dram_bw) * 1e6 # 1250 us
        du0_A_done = time_dram_du_A # 1250
        du1_A_start = max(interval_A, du0_A_done) # max(10, 1250) = 1250
        du1_A_done = du1_A_start + time_dram_du_A # 1250 + 1250 = 2500

        # T_B (Intermediate), no interval, triggered by T_A's completion
        time_transfer_A_to_B = 0 # Same resource
        du_size_B = 25.0; time_dram_du_B = (du_size_B / dram_bw) * 1e6 # 1250 us
        du0_B_start = du0_A_done + time_transfer_A_to_B # 1250
        du0_B_done = du0_B_start + time_dram_du_B # 2500
        du1_B_start = max(du1_A_done + time_transfer_A_to_B, du0_B_done) # max(2500, 2500) = 2500
        du1_B_done = du1_B_start + time_dram_du_B # 2500 + 1250 = 3750

        # T_C (Compute)
        time_transfer_B_to_C_du = (du_size_B / noc_bw) * 1e6 # 250 us
        time_compute_du = 20.0 / 2 # 10 us
        du0_C_arrival = du0_B_done + time_transfer_B_to_C_du # 2500 + 250 = 2750
        du0_C_done = du0_C_arrival + time_compute_du # 2760
        du1_C_arrival = du1_B_done + time_transfer_B_to_C_du # 3750 + 250 = 4000
        du1_C_start = max(du1_C_arrival, du0_C_done) # max(4000, 2760) = 4000
        expected_makespan = du1_C_start + time_compute_du # 4010
        
        print(f"--- Theoretical Calculation (Corrected Interval Logic) ---")
        print(f"  - T_B DU-0 starts its work at: {du0_B_start:.2f} us (driven by T_A completion)")
        print(f"  - T_B DU-1 starts its work at: {du1_B_start:.2f} us (driven by T_A completion and resource availability)")
        print(f"  Expected Makespan: {expected_makespan:.2f} us")
        
        print("\n--- Simulation Log ---")
        router = GlobalRouter(network, links)
        simulator = Simulator(dag, placement, network, links, router)
        simulated_makespan = simulator.start_simulation()

        print(f"\n--- Simulation Result ---")
        print(f"  Simulated Makespan: {simulated_makespan:.2f} us")

        tolerance = 0.01 
        if math.isclose(expected_makespan, simulated_makespan, rel_tol=tolerance):
            print("\n[SUCCESS] Simulated makespan matches the expected value within tolerance.")
        else:
            print(f"\n[FAILURE] Simulated makespan ({simulated_makespan:.2f}) differs significantly from expected value ({expected_makespan:.2f}).")
        print("="*50)
        
    run_simulator_verification_test()