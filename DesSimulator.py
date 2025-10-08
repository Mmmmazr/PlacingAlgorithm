import simpy
from typing import Dict, List
from collections import defaultdict
from BasicDefinitions import Task, DPU, Link, Placement, Resource
import math

# ... GlobalRouter and SharedBus classes remain unchanged ...
class GlobalRouter:
    def __init__(self, network: Dict[str, DPU], links: Dict[str, Link]):
        self.resources: List[str] = [res_id for dpu in network.values() for res_id in dpu.resources]
        self.res_map = {name: i for i, name in enumerate(self.resources)}
        self.num_res = len(self.resources)
        self.dist = [[float('inf')] * self.num_res for _ in range(self.num_res)]
        self.next_hop = [[-1] * self.num_res for _ in range(self.num_res)]
        for i in range(self.num_res): self.dist[i][i] = 0; self.next_hop[i][i] = i
        for dpu in network.values():
            for u_name, v_name in dpu.noc:
                if u_name in self.res_map and v_name in self.res_map:
                    u, v = self.res_map[u_name], self.res_map[v_name]
                    self.dist[u][v] = 1; self.dist[v][u] = 1; self.next_hop[u][v] = v; self.next_hop[v][u] = u
        for link in links.values():
            u_nic, v_nic = f"{link.source_dpu}_nic", f"{link.dest_dpu}_nic"
            if u_nic in self.res_map and v_nic in self.res_map:
                u, v = self.res_map[u_nic], self.res_map[v_nic]
                self.dist[u][v] = 1; self.dist[v][u] = 1; self.next_hop[u][v] = v; self.next_hop[v][u] = u
        self._floyd_warshall()
    def _floyd_warshall(self):
        for k in range(self.num_res):
            for i in range(self.num_res):
                for j in range(self.num_res):
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]; self.next_hop[i][j] = self.next_hop[i][k]
    def get_path(self, source_res_id: str, dest_res_id: str) -> List[str]:
        if source_res_id == dest_res_id: return [source_res_id]
        if source_res_id not in self.res_map or dest_res_id not in self.res_map: return []
        u, v = self.res_map[source_res_id], self.res_map[dest_res_id]
        if self.next_hop[u][v] == -1: return []
        path_indices = [u]
        while u != v: u = self.next_hop[u][v]; path_indices.append(u)
        return [self.resources[i] for i in path_indices]

class SharedBus:
    def __init__(self, env, name, bandwidth_MBps):
        self.env = env; self.name = name; self.bandwidth_MB_us = bandwidth_MBps / 1e6
        self.active_transfers = []; self.process = env.process(self.run()); self.wakeup_event = env.event()
        self.time_of_last_update = 0.0; self.num_active_at_last_update = 0
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
                if t['remaining'] > 1e-9: new_active_list.append(t)
                else:
                    t['remaining'] = 0
                    if not t['event'].triggered: t['event'].succeed()
            self.active_transfers = new_active_list
            self.time_of_last_update = self.env.now
            self.num_active_at_last_update = len(self.active_transfers)
            try:
                if not self.active_transfers:
                    yield self.wakeup_event
                    self.wakeup_event = self.env.event()
                    continue
                effective_bw = self.bandwidth_MB_us / len(self.active_transfers)
                if effective_bw <= 1e-12: yield self.env.timeout(1e9); continue
                times_to_finish = [t['remaining'] / effective_bw for t in self.active_transfers]
                time_to_next_completion = max(0, min(times_to_finish))
                yield self.env.timeout(time_to_next_completion)
            except simpy.Interrupt: pass
    def transfer(self, data_size_mb):
        if data_size_mb <= 1e-9: return self.env.timeout(0)
        completion_event = self.env.event(); was_idle = (len(self.active_transfers) == 0)
        self.active_transfers.append({'remaining': data_size_mb, 'event': completion_event})
        if not self.process.is_alive: self.process = self.env.process(self.run())
        if was_idle:
            if not self.wakeup_event.triggered: self.wakeup_event.succeed()
        else: self.process.interrupt()
        return completion_event

class Simulator:
    def __init__(self, dag: Dict[str, Task], placement: Placement, network: Dict[str, DPU], links: Dict[str, Link], router: GlobalRouter):
        self.dag = dag; self.placement = placement; self.network = network; self.links = links; self.router = router
        self._res_to_dpu_map = {}
        for dpu in self.network.values():
            for res_id in dpu.resources: self._res_to_dpu_map[res_id] = dpu.id
            
    def _get_resource_dpu_id(self, resource_id: str) -> str:
        return self._res_to_dpu_map.get(resource_id)
    
    def start_simulation(self):
        self.env = simpy.Environment()
        self.resources = {}
        NOC_BANDWIDTH_MBPS = 100_000
        for dpu_id, dpu in self.network.items():
            self.resources[f"{dpu_id}_noc_bus"] = SharedBus(self.env, f"{dpu_id}_noc_bus", NOC_BANDWIDTH_MBPS)
            for res in dpu.resources.values():
                if res.type == 'compute': self.resources[res.id] = simpy.Resource(self.env, capacity=res.capacity)
                elif res.bandwidth_mbps > 0: self.resources[res.id] = SharedBus(self.env, res.id, res.bandwidth_mbps)
        for link in self.links.values():
            bandwidth_MBps = link.bandwidth_gbps * 125
            link_key = tuple(sorted((f"{link.source_dpu}_nic", f"{link.dest_dpu}_nic")))
            self.resources[link_key] = SharedBus(self.env, link.id, bandwidth_MBps)
        self.task_du_done_events = defaultdict(list)
        for task_id, task in self.dag.items():
            num_dus = task.du_num if task.du_num > 0 else 1
            self.task_du_done_events[task_id] = [self.env.event() for _ in range(num_dus)]
        self.du_arrival_stores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        self.env.run(self.run())
        return self.env.now
    
    def run(self):
        for task_id in self.dag: self.env.process(self._task_process_router(task_id))
        final_tasks = [tid for tid, task in self.dag.items() if not task.children]
        if not final_tasks: return self.env.timeout(0)
        final_events = [ev for tid in final_tasks for ev in self.task_du_done_events[tid]]
        return simpy.AllOf(self.env, final_events)
    
    def _task_process_router(self, task_id: str):
        task = self.dag[task_id]
        if task.type == 'compute': yield self.env.process(self._master_process(task_id, self._compute_worker))
        elif task.type == 'data': yield self.env.process(self._master_process(task_id, self._data_worker))

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
            for i in range(parent_du_num):
                self.env.process(self._data_transfer_worker_du(parent_id, task_id, i))
        for i in range(child_du_num):
            self.env.process(worker_func(task_id, i, pipelined_parents, non_pipelined_parents))
        yield simpy.AllOf(self.env, self.task_du_done_events[task_id])

    def _compute_worker(self, task_id: str, du_index: int, pipelined_parents: List[str], non_pipelined_parents: List[str]):
        task = self.dag[task_id]
        yield self.env.process(self._wait_for_parent_data(task_id, du_index, pipelined_parents, non_pipelined_parents))
        compute_res_id = self.placement[task_id]
        compute_resource = self.resources[compute_res_id]
        
        # --- 【已修正】恢复对 task.workload 的检查 ---
        if hasattr(task, 'workload'):
            du_compute_time = task.workload / (task.du_num or 1)
        else:
            workload = {'linear': 10, 'slice': 2, 'rope': 15, 'view': 1, 'einsum': 25, 'add': 2, 'softmax': 8}
            base_workload = workload.get(task.compute_type, 5)
            du_compute_time = base_workload / (task.du_num or 1)
        
        with compute_resource.request() as req:
            yield req
            if du_compute_time > 0:
                yield self.env.timeout(du_compute_time)
        self.task_du_done_events[task_id][du_index].succeed()

    def _data_worker(self, task_id: str, du_index: int, pipelined_parents: List[str], non_pipelined_parents: List[str]):
        task = self.dag[task_id]
        
        # --- 【已修正】恢复对 du_interval_us 的处理 ---
        du_interval = getattr(task, 'du_interval_us', 0)
        if du_interval > 0:
            start_delay = du_index * du_interval
            if start_delay > 0:
                yield self.env.timeout(start_delay)

        yield self.env.process(self._wait_for_parent_tasks(task_id, du_index, pipelined_parents, non_pipelined_parents))
        storage_res_id = self.placement[task_id]
        storage_bus = self.resources.get(storage_res_id)
        du_size = task.du_size if task.du_size > 0 else task.data_size / (task.du_num or 1)
        if du_size > 0 and storage_bus:
            yield storage_bus.transfer(du_size)
        self.task_du_done_events[task_id][du_index].succeed()

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
        if wait_for_events: yield simpy.AllOf(self.env, wait_for_events)

    def _wait_for_parent_tasks(self, task_id: str, du_index: int, pipelined_parents: List[str], non_pipelined_parents: List[str]):
        wait_for_events = []
        for parent_id in pipelined_parents:
            wait_for_events.append(self.task_du_done_events[parent_id][du_index])
        for parent_id in non_pipelined_parents:
            wait_for_events.extend(self.task_du_done_events[parent_id])
        if wait_for_events: yield simpy.AllOf(self.env, wait_for_events)

    def _data_transfer_worker_du(self, source_id: str, dest_id: str, du_index: int):
        source_task = self.dag[source_id]
        du_size = source_task.du_size if source_task.du_size > 0 else source_task.data_size / (source_task.du_num or 1)
        if du_size <= 1e-9:
            if self.du_arrival_stores[source_id][dest_id][du_index] is None:
                self.du_arrival_stores[source_id][dest_id][du_index] = simpy.Store(self.env, capacity=1)
            yield self.du_arrival_stores[source_id][dest_id][du_index].put(True)
            return
        yield self.task_du_done_events[source_id][du_index]
        source_res_id = self.placement[source_id]; dest_res_id = self.placement[dest_id]
        comm_path = self.router.get_path(source_res_id, dest_res_id)
        for j in range(len(comm_path) - 1):
            u_res, v_res = comm_path[j], comm_path[j+1]
            u_dpu = self._get_resource_dpu_id(u_res); v_dpu = self._get_resource_dpu_id(v_res)
            bus_resource = None
            if u_dpu == v_dpu: bus_resource = self.resources.get(f"{u_dpu}_noc_bus")
            else:
                link_key = tuple(sorted((f"{u_dpu}_nic", f"{v_dpu}_nic")))
                bus_resource = self.resources.get(link_key)
            if bus_resource: yield bus_resource.transfer(du_size)
            elif u_dpu != v_dpu: print(f"Warning: No bus resource found for transfer from {u_res} to {v_res}")
        if self.du_arrival_stores[source_id][dest_id][du_index] is None:
            self.du_arrival_stores[source_id][dest_id][du_index] = simpy.Store(self.env, capacity=1)
        yield self.du_arrival_stores[source_id][dest_id][du_index].put(True)


if __name__ == '__main__':
    # This test should now pass with the corrected Simulator class
    def run_simulator_verification_test():
        print("\n" + "="*50)
        print("Running Standalone Verification Test for DesSimulator (Pipelining Fixed)")
        print("="*50)

        DU_NUM = 2; DATA_SIZE = 100.0; DU_SIZE = DATA_SIZE / DU_NUM; DU_INTERVAL = 10.0
        dag: Dict[str, Task] = {
            "T_A": Task(id="T_A", name="LoadData", type='data', data_size=DATA_SIZE, du_size=DU_SIZE, du_num=DU_NUM, children=["T_B"]),
            "T_B": Task(id="T_B", name="Process", type='compute', data_size=0, du_size=0, du_num=DU_NUM, compute_type='linear', parents=["T_A"])
        }
        dag["T_A"].du_interval_us = DU_INTERVAL
        # Crucially, set the workload attribute that the test expects
        WORKLOAD_T_B = 50.0
        setattr(dag["T_B"], 'workload', WORKLOAD_T_B)

        dram_bw = 20_000; noc_bw = 100_000; link_bw_gbps = 100; link_bw_MBps = link_bw_gbps * 125
        network: Dict[str, DPU] = {
            "dpu1": DPU("dpu1", resources={"dpu1_dram": Resource("dpu1_dram", "dram", "storage", bandwidth_mbps=dram_bw), "dpu1_nic": Resource("dpu1_nic", "nic", "communication")}, noc=[("dpu1_dram", "dpu1_nic")]),
            "dpu2": DPU("dpu2", resources={"dpu2_arm": Resource("dpu2_arm", "arm", "compute", capacity=1), "dpu2_nic": Resource("dpu2_nic", "nic", "communication")}, noc=[("dpu2_arm", "dpu2_nic")])
        }
        links: Dict[str, Link] = {"link_dpu1_dpu2": Link("link_dpu1_dpu2", "dpu1", "dpu2", link_bw_gbps)}
        placement: Placement = {"T_A": "dpu1_dram", "T_B": "dpu2_arm"}

        # --- Theoretical Calculation ---
        time_dram_du = (DU_SIZE / dram_bw) * 1e6 # 2500 us
        
        du0_read_start = 0
        du1_read_start = du0_read_start + DU_INTERVAL # 10
        
        # DRAM is a shared bus, they will contend
        # From t=0 to t=10, only du0 is active. Data transferred: 10 * dram_bw/1 = 0.2 MB
        # From t=10 onwards, both are active. Effective bw for each = dram_bw/2
        du0_rem_data = DU_SIZE - 0.2 # 49.8 MB
        time_for_du0_to_finish_contended = (du0_rem_data / (dram_bw/2)) * 1e6 # 4980 us
        du0_read_done = du1_read_start + time_for_du0_to_finish_contended # 10 + 4980 = 4990
        
        # When du0 finishes, du1 has also been running for 4980us. Data transferred for du1: 49.8 MB
        du1_rem_data = DU_SIZE - 49.8 # 0.2 MB
        # Now du1 gets full bandwidth
        time_for_du1_to_finish_uncontended = (du1_rem_data / dram_bw) * 1e6 # 10 us
        du1_read_done = du0_read_done + time_for_du1_to_finish_uncontended # 4990 + 10 = 5000

        time_noc = (DU_SIZE / noc_bw) * 1e6 # 500 us
        time_link_uncontended = (DU_SIZE / link_bw_MBps) * 1e6 # 4000 us
        time_compute_du = WORKLOAD_T_B / DU_NUM # 25 us

        # The link is the main bottleneck where DUs will overlap
        du0_link_start = du0_read_done + time_noc # 4990 + 500 = 5490
        du1_link_start = du1_read_done + time_noc # 5000 + 500 = 5500
        
        # From t=5490 to t=5500, only du0 on link. Data transferred: 10 * link_bw_MBps/1 = 0.125 MB
        du0_rem_link_data = DU_SIZE - 0.125 # 49.875 MB
        # From t=5500, both are on link. Effective bw for each = link_bw_MBps/2
        time_for_du0_to_finish_link_contended = (du0_rem_link_data / (link_bw_MBps/2)) * 1e6 # 7980 us
        du0_link_done = du1_link_start + time_for_du0_to_finish_link_contended # 5500 + 7980 = 13480
        
        du1_rem_link_data = DU_SIZE - (time_for_du0_to_finish_link_contended * (link_bw_MBps/2) / 1e6) # 0.125 MB
        time_for_du1_to_finish_link_uncontended = (du1_rem_link_data / link_bw_MBps) * 1e6 # 10 us
        du1_link_done = du0_link_done + time_for_du1_to_finish_link_uncontended # 13480 + 10 = 13490

        du0_cpu_arrival = du0_link_done + time_noc # 13480 + 500 = 13980
        du1_cpu_arrival = du1_link_done + time_noc # 13490 + 500 = 13990

        du0_compute_finish = du0_cpu_arrival + time_compute_du # 13980 + 25 = 14005
        du1_compute_start = max(du1_cpu_arrival, du0_compute_finish) # max(13990, 14005) = 14005
        expected_makespan = du1_compute_start + time_compute_du # 14005 + 25 = 14030

        print(f"--- Theoretical Calculation (Contention Aware & Pipelined) ---")
        print(f"  - DU-0 finishes link at: {du0_link_done:.2f} us")
        print(f"  - DU-1 finishes link at: {du1_link_done:.2f} us")
        print(f"  - DU-0 compute finishes at: {du0_compute_finish:.2f} us")
        print(f"  - DU-1 compute starts at: {du1_compute_start:.2f} us")
        print(f"  ---------------------------------")
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