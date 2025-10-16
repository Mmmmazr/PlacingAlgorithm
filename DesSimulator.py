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
        self.bandwidth_MB_us = bandwidth_MBps / 1e6 # 所有的时间单位都以us计算
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
                for t in self.active_transfers: 
                    if not t.get('is_new', False):
                        t['remaining'] -= data_transferred

            new_active_list = []
            for t in self.active_transfers:
                if t['remaining'] > 1e-9: 
                    t['is_new'] = False
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
        
        # print(f"[{self.env.now:8.2f}] [Bus: {self.name}] New transfer request for {data_size_mb:.2f} MB.")

        completion_event = self.env.event()
        was_idle = (len(self.active_transfers) == 0)
        self.active_transfers.append({'remaining': data_size_mb, 'event': completion_event, 'is_new': True})
        if not self.process.is_alive: 
            self.process = self.env.process(self.run())
        if was_idle: #空
            if not self.wakeup_event.triggered: 
                self.wakeup_event.succeed()
        else: #非空
            self.process.interrupt()
        return completion_event

class NewContainer(simpy.Container):
    def put(self, amount: float):
        if amount <= 1e-9:
            raise ValueError
        if amount > self.capacity:
            raise ValueError
        if self.level + amount > self.capacity:
            raise ValueError
        return super().put(amount)
class StorageResource:
    def __init__(self, env, name, bandwidth_mbps, memory_gb):
        self.bus = SharedBus(env, f"{name}_bus", bandwidth_mbps)
        self.memory_container = NewContainer(env, capacity=memory_gb * 1024, init=0)
        # capacity被从GB->MB
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
        from BasicDefinitions import NOC_BANDWIDTH_MBPS
        for dpu_id, dpu in self.network.items():
            self.resources[f"{dpu_id}_noc_bus"] = SharedBus(self.env, f"{dpu_id}_noc_bus", NOC_BANDWIDTH_MBPS)
            for res in dpu.resources.values():
                if res.type == 'compute': 
                    core_pool = simpy.Store(self.env, capacity=res.capacity)
                    for i in range(res.capacity):
                        core_pool.put(f"core_{i}")
                    self.resources[res.id] = core_pool
                elif res.type == 'storage' and res.bandwidth_MBps > 0: 
                    self.resources[res.id] = StorageResource(self.env, res.id, res.bandwidth_MBps, res.memory)
        
        for link in self.links.values():
            bandwidth_MBps = link.bandwidth_MBps
            link_key = tuple(sorted((f"{link.source_dpu}_nic", f"{link.dest_dpu}_nic"))) # 保证link_key唯一
            self.resources[link_key] = SharedBus(self.env, link.id, bandwidth_MBps)
            
        self.task_du_done_events = defaultdict(list)
        for task_id, task in self.dag.items():
            num_dus = task.du_num if task.du_num > 0 else 1
            self.task_du_done_events[task_id] = [self.env.event() for _ in range(num_dus)]

        self.du_arrival_stores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None))) # ???
        try:
            self.env.run(self.run())
            return self.env.now
        except ValueError as e: # 通过检测是否超出存储资源限制，给模拟退火一个信号
            # print(e)
            return float('inf') 
    
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
            if task.children:
                du_num = task.du_num if task.du_num > 0 else 1
                for i in range(du_num):
                    self.env.process(self._memory_free(task_id, i))

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

        # print(f"[{self.env.now:8.2f}] [ComputeWorker] Task '{task.id}' DU-{du_index}: Waiting for parent data.")

        yield self.env.process(self._wait_for_parent_data(task_id, du_index, pipelined_parents, non_pipelined_parents))
        
        # print(f"[{self.env.now:8.2f}] [ComputeWorker] Task '{task.id}' DU-{du_index}: Parent data arrived. Requesting compute resource.")
        
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
            # print(f"[{self.env.now:8.2f}] [ComputeWorker] Task '{task.id}' DU-{du_index}: Got core, starting compute on '{compute_res_id}' for {du_compute_time:.2f} us.")
            if du_compute_time > 0:
                yield self.env.timeout(du_compute_time)
        finally:
            # print(f"[{self.env.now:8.2f}] [ComputeWorker] Task '{task.id}' DU-{du_index}: Computation done, returning core.")
            yield compute_resource_pool.put(core_token) # 归还核心

        self.task_du_done_events[task_id][du_index].succeed()

        # print(f"[{self.env.now:8.2f}] [Event] Task '{task.id}' DU-{du_index} finished. Event succeeded.")

    def _data_worker(self, task_id: str, du_index: int, pipelined_parents: List[str], non_pipelined_parents: List[str]):
        task = self.dag[task_id]
        if not task.parents:
            du_interval = getattr(task, 'du_interval', 10)
            if du_interval > 0 and du_index > 0:
                yield self.env.timeout(du_index * du_interval)
                # print(f"[{self.env.now:8.2f}] [DataWorker] Task '{task.id}' DU-{du_index}: start transfer.")

        # if task.parents:
        #     print(f"[{self.env.now:8.2f}] [DataWorker] Task '{task.id}' DU-{du_index}: Waiting for parent data.")

        yield self.env.process(self._wait_for_parent_data(task_id, du_index, pipelined_parents, non_pipelined_parents))

        # if task.parents:
        #     print(f"[{self.env.now:8.2f}] [DataWorker] Task '{task.id}' DU-{du_index}: Parent data arrived.")

        storage_res_id = self.placement[task_id]
        storage_resource = self.resources.get(storage_res_id)
        du_size = task.du_size if task.du_size > 0 else (task.data_size / (task.du_num or 1))
        if du_size > 0 and storage_resource:

            # print(f"[{self.env.now:8.2f}] [DataWorker] Task '{task.id}' DU-{du_index}: Starting storage op ({du_size:.2f} MB) on '{storage_res_id}'.")
            
            yield storage_resource.bus.transfer(du_size)
            yield storage_resource.memory_container.put(du_size)
            
            # print(f"[{self.env.now:8.2f}] [DataWorker] Task '{task.id}' DU-{du_index} finished storage op, allocating {du_size:.2f} MB memory.")
        
        self.task_du_done_events[task_id][du_index].succeed()

        # print(f"[{self.env.now:8.2f}] [Event] Task '{task.id}' DU-{du_index} finished. Event succeeded.")

    def _memory_free(self, task_id: str, du_index: int):
        '''释放当前任务task_id占用的内存'''
        task = self.dag[task_id]
        storage_res_id = self.placement[task_id]
        du_size = task.du_size if task.du_size > 0 else (task.data_size / (task.du_num or 1))

        if not task.children or du_size <= 1e-9:
            return

        child_completion_events = []
        for child_id in task.children:
            child_task = self.dag[child_id]
            child_du_num = child_task.du_num if child_task.du_num > 0 else 1
            if child_du_num == task.du_num:
                 child_completion_events.append(self.task_du_done_events[child_id][du_index])
            else: # 不等，可能child需要所有du，要等待子任务所有du都完成
                child_completion_events.extend(self.task_du_done_events[child_id])

        unique_events = list(set(child_completion_events)) # 去重，如果所有子任务都需要父任务的所有du，child_completion_event有很多重复
        yield simpy.AllOf(self.env, unique_events)
        
        # print(f"[{self.env.now:8.2f}] [Memory Free] All children consumed '{task.id}'-DU{du_index}, freeing {du_size:.2f} MB from '{storage_res_id}'.")
        
        storage_resource = self.resources.get(storage_res_id)
        if storage_resource:
            yield storage_resource.memory_container.get(du_size)

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
        
        # print(f"[{self.env.now:8.2f}] [Transfer] Waiting for source '{source_id}'-DU{du_index} to be ready.")

        yield self.task_du_done_events[source_id][du_index]

        # print(f"[{self.env.now:8.2f}] [Transfer] Source '{source_id}'-DU{du_index} is ready. Starting transfer to '{dest_id}'.")

        source_res_id = self.placement[source_id]
        dest_res_id = self.placement[dest_id]
        comm_path = self.router.get_path(source_res_id, dest_res_id)

        # print(f"[{self.env.now:8.2f}] [Transfer] Path for '{source_id}'-DU{du_index} -> '{dest_id}': {' -> '.join(comm_path)}")

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
                
                # print(f"[{self.env.now:8.2f}] [Transfer] DU-{du_index} ({du_size:.2f} MB) entering bus from '{u_res}' to '{v_res}'.")
                
                yield bus_resource.transfer(du_size)

                # print(f"[{self.env.now:8.2f}] [Transfer] DU-{du_index} ({du_size:.2f} MB) exited bus from '{u_res}' to '{v_res}'.")
            
            elif u_dpu != v_dpu: 
                print(f"Warning: No bus resource found for transfer from {u_res} to {v_res}")
        
        if self.du_arrival_stores[source_id][dest_id][du_index] is None:
            self.du_arrival_stores[source_id][dest_id][du_index] = simpy.Store(self.env, capacity=1)
        
        # print(f"[{self.env.now:8.2f}] [Transfer] DU-{du_index} from '{source_id}' has arrived at destination for '{dest_id}'.")
        
        yield self.du_arrival_stores[source_id][dest_id][du_index].put(True)

if __name__ == '__main__':
    # 导入所需的创建函数
    from DpuNetwork import create_dpu_network
    from TaskGraph import create_workflow_dag
    
    def run_simulator_verification_test():

        network, links = create_dpu_network(r"C:\code\PlacingAlgorithm\test_cases\DpuNetwork1.json")
        dag = create_workflow_dag(r"C:\code\PlacingAlgorithm\test_cases\TaskGraph1.json")

        setattr(dag["T_B"], 'workload', 20.0) # 每个DU计算10us
        setattr(dag["T_D"], 'workload', 30.0) # 每个DU计算15us
        setattr(dag['T_A'], 'du_interval', 10.0) # source data间隔10us发出

        placement: Placement = {
            "T_A": "dpu1_dram",
            "T_B": "dpu1_arm",
            "T_C": "dpu1_dram",
            "T_D": "dpu2_arm"
        }
        
        print("\n--- Log ---")
        router = GlobalRouter(network, links)
        simulator = Simulator(dag, placement, network, links, router)
        simulated_makespan = simulator.start_simulation()

        print("\n--- Result ---")
        print(f" {simulated_makespan:.2f} us")

    run_simulator_verification_test()