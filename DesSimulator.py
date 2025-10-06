import simpy
from typing import Dict, List
from collections import defaultdict
from BasicDefinitions import Task, DPU, Link, Placement

class GlobalRouter:
    """
    一个全局路由器，计算网络中任意两个资源之间的最短路径。
    图的节点是所有DPU中的所有可调度资源（包括展开的计算核心）。
    图的边包括DPU内部的NoC连接和DPU之间的外部链路。
    """
    def __init__(self, network: Dict[str, DPU], links: Dict[str, Link]):
        self.resources: List[str] = []
        # 将计算资源展开为独立的可调度单元
        for dpu in network.values():
            for res_id, res_obj in dpu.resources.items():
                if res_obj.type == 'compute':
                    # for i in range(res_obj.capacity):
                        # self.resources.append(f"{res_id}_{i}")
                    self.resources.append(res_id)
                else:
                    self.resources.append(res_id)

        self.res_map = {name: i for i, name in enumerate(self.resources)}
        self.num_res = len(self.resources)
        self.dist = [[float('inf')] * self.num_res for _ in range(self.num_res)]
        self.next_hop = [[-1] * self.num_res for _ in range(self.num_res)]

        for i in range(self.num_res):
            self.dist[i][i] = 0
            self.next_hop[i][i] = i

        # 1. 添加DPU内部NoC连接
        for dpu in network.values():
            for u_base_name, v_base_name in dpu.noc:
                u_res_obj = dpu.resources[u_base_name]
                v_res_obj = dpu.resources[v_base_name]

                u_names = [u_base_name]
                v_names = [v_base_name]

                for u_name in u_names:
                    for v_name in v_names:
                        if u_name in self.res_map and v_name in self.res_map:
                            u, v = self.res_map[u_name], self.res_map[v_name]
                            print(dpu.id, u_name, v_name)
                            self.dist[u][v] = 1
                            self.dist[v][u] = 1
                            self.next_hop[u][v] = v
                            self.next_hop[v][u] = u

        # 2. 添加DPU之间通过NIC的连接
        for link in links.values():
            u_nic = f"{link.source_dpu}_nic"
            v_nic = f"{link.dest_dpu}_nic"
            if u_nic in self.res_map and v_nic in self.res_map:
                u, v = self.res_map[u_nic], self.res_map[v_nic]
                print(u_nic, v_nic)
                self.dist[u][v] = 1
                self.dist[v][u] = 1
                self.next_hop[u][v] = v
                self.next_hop[v][u] = u
        
        print("GlobalRouter: Initialized with all schedulable resources.")
        self._floyd_warshall()
        print("GlobalRouter: Floyd-Warshall calculation finished.")

    def _floyd_warshall(self):
        for k in range(self.num_res):
            for i in range(self.num_res):
                for j in range(self.num_res):
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]
                        self.next_hop[i][j] = self.next_hop[i][k]
        print(self.dist[1][2])

    def get_path(self, source_res_id: str, dest_res_id: str) -> List[str]:
        if source_res_id == dest_res_id: return [source_res_id]
        if source_res_id not in self.res_map or dest_res_id not in self.res_map: return []
        u, v = self.res_map[source_res_id], self.res_map[dest_res_id]
        if self.next_hop[u][v] == -1: return []
        path_indices = [u]
        while u != v:
            u = self.next_hop[u][v]
            path_indices.append(u)
        return [self.resources[i] for i in path_indices]

    def print_sample_paths(self):
        """(Verification Method) 打印一些示例路径来检查路由器的正确性。"""
        print("\n--- GlobalRouter Sample Path Verification ---")
        
        # *** FIX: Querying for specific, expanded resource IDs that actually exist in the graph ***
        # 路径 1: DPU内部 (dram -> 一个特定的arm core)
        p1_start, p1_end = "dpu1_dram", "dpu1_arm"
        path1 = self.get_path(p1_start, p1_end)
        print(f"Path from {p1_start} to {p1_end}: {path1}")

        # 路径 2: DPU之间 (一个特定的arm core -> ssd)
        p2_start, p2_end = "dpu1_arm", "dpu2_ssd"
        path2 = self.get_path(p2_start, p2_end)
        print(f"Path from {p2_start} to {p2_end}: {path2}")
        
        # 路径 3: DPU之间 (一个特定的dpa core -> 另一个特定的dpa core)
        p3_start, p3_end = "dpu3_dpa", "dpu4_dpa"
        path3 = self.get_path(p3_start, p3_end)
        print(f"Path from {p3_start} to {p3_end}: {path3}")


class SharedBus:
    def __init__(self, env, name, bandwidth_mbps):
        self.env = env
        self.name = name
        self.bandwidth_mb_us = (bandwidth_mbps / 8) / 1e6
        self.active_transfers = []
        self.process = env.process(self.run())
        self.wakeup_event = env.event()
    def run(self):
        time_of_last_update = self.env.now
        while True:
            if not self.active_transfers:
                yield self.wakeup_event
                self.wakeup_event = self.env.event()
                time_of_last_update = self.env.now
                continue
            try:
                effective_bw = self.bandwidth_mb_us / len(self.active_transfers)
                if effective_bw <= 1e-12:
                    yield self.env.timeout(1e9)
                    continue
                times_to_finish = [t['remaining'] / effective_bw for t in self.active_transfers]
                time_to_next_completion = min(times_to_finish)
                yield self.env.timeout(time_to_next_completion)
                time_passed = self.env.now - time_of_last_update
                data_transferred = effective_bw * time_passed
                new_active_list = []
                for t in self.active_transfers:
                    t['remaining'] -= data_transferred
                    if t['remaining'] > 1e-9: new_active_list.append(t)
                    else: t['event'].succeed()
                self.active_transfers = new_active_list
                time_of_last_update = self.env.now
            except simpy.Interrupt:
                time_passed = self.env.now - time_of_last_update
                num_transfers_before_interrupt = len(self.active_transfers) - 1
                if num_transfers_before_interrupt > 0:
                    old_effective_bw = self.bandwidth_mb_us / num_transfers_before_interrupt
                    data_transferred = old_effective_bw * time_passed
                    for t in self.active_transfers[:-1]: t['remaining'] -= data_transferred
                time_of_last_update = self.env.now
    def transfer(self, data_size_mb):
        if data_size_mb <= 1e-9: return self.env.timeout(0)
        completion_event = self.env.event()
        was_idle = (len(self.active_transfers) == 0)
        self.active_transfers.append({'remaining': data_size_mb, 'event': completion_event})
        if was_idle:
            if not self.wakeup_event.triggered: self.wakeup_event.succeed()
        else:
            self.process.interrupt()
        return completion_event

class Simulator:
    def __init__(self, dag: Dict[str, Task], placement: Placement,
                 network: Dict[str, DPU], links: Dict[str, Link], router: GlobalRouter):
        self.env = simpy.Environment()
        self.dag = dag
        self.placement = placement
        self.network = network
        self.links = links
        self.router = router
        self.resources = {}
        self._res_to_dpu_map = {}

        # 1. Initialize all physical resources and create SharedBuses for NoC/Links
        # DPU-internal NoC buses (simplified as one bus per DPU)
        for dpu_id, dpu in network.items():
            for res in dpu.resources.values():
                self._res_to_dpu_map[res.id] = dpu_id
                if res.type == 'compute':
                    for i in range(res.capacity):
                        self.resources[f"{res.id}_{i}"] = simpy.Resource(self.env, capacity=1)
                        self._res_to_dpu_map[f"{res.id}_{i}"] = dpu_id
                elif res.bandwidth_mbps > 0:
                    self.resources[res.id] = SharedBus(self.env, res.id, res.bandwidth_mbps)
        
        # External network link buses
        for link in links.values():
            bandwidth_mbps = link.bandwidth_gbps * 1000 / 8
            # The resource ID for a link is the NIC-to-NIC connection
            link_res_id_fwd = (f"{link.source_dpu}_nic", f"{link.dest_dpu}_nic")
            self.resources[link_res_id_fwd] = SharedBus(self.env, link.id, bandwidth_mbps)

        # ... (rest of init is unchanged) ...
        self.task_du_done_events: Dict[str, List[simpy.Event]] = {}
        for task_id, task in dag.items():
            num_dus = task.du_num if task.du_num > 0 else 1
            self.task_du_done_events[task_id] = [self.env.event() for _ in range(num_dus)]
        self.du_arrival_stores = defaultdict(lambda: defaultdict(dict))

    def _get_resource_dpu_id(self, resource_id: str) -> str:
        """Optimized lookup for resource's DPU ID."""
        dpu_id = self._res_to_dpu_map.get(resource_id)
        if dpu_id:
            return dpu_id
        raise ValueError(f"Resource ID {resource_id} not found in any DPU.")

    # ... (run, start_simulation, _task_process_router, _compute_*, _data_* methods are unchanged) ...
    def run(self):
        for task_id in self.dag: self.env.process(self._task_process_router(task_id))
        final_tasks = [tid for tid, task in self.dag.items() if not task.children]
        if not final_tasks: return self.env.timeout(0)
        final_events = [ev for tid in final_tasks for ev in self.task_du_done_events[tid]]
        return simpy.AllOf(self.env, final_events)
    def start_simulation(self):
        self.env.run(self.run())
        return self.env.now
    def _task_process_router(self, task_id: str):
        task = self.dag[task_id]
        if task.type == 'compute': yield self.env.process(self._compute_task_process(task_id))
        elif task.type == 'data': yield self.env.process(self._data_task_process(task_id))
    def _compute_task_process(self, task_id: str):
        task = self.dag[task_id]
        num_dus = task.du_num if task.du_num > 0 else 1
        for parent_id in task.parents: self.env.process(self._data_transfer_worker(source_id=parent_id, dest_id=task_id))
        for i in range(num_dus): self.env.process(self._compute_worker(task_id, i))
        yield simpy.AllOf(self.env, self.task_du_done_events[task_id])
    def _compute_worker(self, task_id: str, du_index: int):
        task = self.dag[task_id]
        arrival_gets = []
        for parent_id in task.parents:
            if du_index not in self.du_arrival_stores[parent_id][task_id]: self.du_arrival_stores[parent_id][task_id][du_index] = simpy.Store(self.env, capacity=1)
            arrival_gets.append(self.du_arrival_stores[parent_id][task_id][du_index].get())
        if arrival_gets: yield simpy.AllOf(self.env, arrival_gets)
        compute_res_id = self.placement[task_id]
        compute_resource = self.resources[compute_res_id]
        workload = {'linear': 10, 'slice': 2, 'rope': 15, 'view': 1, 'einsum': 25, 'add': 2, 'softmax': 8}
        du_compute_time = workload.get(task.compute_type, 5) / (task.du_num or 1)
        with compute_resource.request() as req:
            yield req
            yield self.env.timeout(du_compute_time)
        self.task_du_done_events[task_id][du_index].succeed()
    def _data_task_process(self, task_id: str):
        task = self.dag[task_id]
        num_dus = task.du_num if task.du_num > 0 else 1
        if not task.parents:
            for i in range(num_dus):
                yield self.env.timeout(0.5)
                self.task_du_done_events[task_id][i].succeed()
            return
        for i in range(num_dus): self.env.process(self._data_worker(task_id, i))
        yield simpy.AllOf(self.env, self.task_du_done_events[task_id])
    def _data_worker(self, task_id: str, du_index: int):
        task = self.dag[task_id]
        parent_du_done_events = [self.task_du_done_events[pid][du_index] for pid in task.parents]
        yield simpy.AllOf(self.env, parent_du_done_events)
        storage_res_id = self.placement[task_id]
        storage_bus = self.resources[storage_res_id]
        du_size = task.data_size / (task.du_num or 1)
        if du_size > 0: yield storage_bus.transfer(du_size)
        self.task_du_done_events[task_id][du_index].succeed()

    def _data_transfer_worker(self, source_id: str, dest_id: str):
        """
        A worker that pipelines all DUs of a source task over the network to a destination task.
        *** MODIFIED: Uses GlobalRouter to get the end-to-end resource path. ***
        """
        source_task = self.dag[source_id]
        num_dus = source_task.du_num if source_task.du_num > 0 else 1
        du_size = source_task.du_size

        if du_size <= 1e-9:
            for i in range(num_dus):
                if i not in self.du_arrival_stores[source_id][dest_id]: self.du_arrival_stores[source_id][dest_id][i] = simpy.Store(self.env, capacity=1)
                yield self.du_arrival_stores[source_id][dest_id][i].put(True)
            return

        source_res_id = self.placement[source_id]
        dest_res_id = self.placement[dest_id]
        
        # Get the full path from the global router
        comm_path = self.router.get_path(source_res_id, dest_res_id)
        
        # Pipeline each DU through the path
        for i in range(num_dus):
            yield self.task_du_done_events[source_id][i]
            
            # Transfer the DU over each segment of the path
            for j in range(len(comm_path) - 1):
                u_res, v_res = comm_path[j], comm_path[j+1]
                u_dpu = self._get_resource_dpu_id(u_res)
                v_dpu = self._get_resource_dpu_id(v_res)
                
                bus_resource = None
                if u_dpu == v_dpu: # Intra-DPU (NoC)
                    # Simplified: Assume transfer is on the destination resource's bus
                    bus_resource = self.resources.get(v_res)
                else: # Inter-DPU (Link)
                    # Find the shared bus for the link between the two NICs
                    link_key = (u_res, v_res)
                    rev_link_key = (v_res, u_res)
                    bus_resource = self.resources.get(link_key) or self.resources.get(rev_link_key)

                if bus_resource:
                    yield bus_resource.transfer(du_size)

            # Signal arrival at the destination
            if i not in self.du_arrival_stores[source_id][dest_id]: self.du_arrival_stores[source_id][dest_id][i] = simpy.Store(self.env, capacity=1)
            yield self.du_arrival_stores[source_id][dest_id][i].put(True)