# placement_algorithm.py

import math
import random
import copy
from typing import Dict, List, Tuple
from BasicDefinitions import Task, Placement, DPU, Link
from DesSimulator import Simulator

#Router使用Floyd-Warshall算法计算两个dpu之间的最短路由
class Router:
    def __init__(self, network: Dict[str, DPU], links: Dict[str, Link]):
        self.dpus = list(network.keys())
        self.dpu_map = {name: i for i, name in enumerate(self.dpus)}
        self.num_dpus = len(self.dpus)
        self.dist = [[float('inf')] * self.num_dpus for _ in range(self.num_dpus)]
        self.next_hop = [[-1] * self.num_dpus for _ in range(self.num_dpus)]

        for i in range(self.num_dpus):
            self.dist[i][i] = 0
            self.next_hop[i][i] = i

        for link in links.values():
            u, v = self.dpu_map[link.source_dpu], self.dpu_map[link.dest_dpu]
            self.dist[u][v] = 1
            self.dist[v][u] = 1
            self.next_hop[u][v] = v
            self.next_hop[v][u] = u
        
        self._floyd_warshall()

    def _floyd_warshall(self):
        for k in range(self.num_dpus):
            for i in range(self.num_dpus):
                for j in range(self.num_dpus):
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]
                        self.next_hop[i][j] = self.next_hop[i][k]

    def get_path(self, source_dpu_id: str, dest_dpu_id: str) -> List[str]:
        if source_dpu_id == dest_dpu_id:
            return []
        u, v = self.dpu_map[source_dpu_id], self.dpu_map[dest_dpu_id]
        if self.next_hop[u][v] == -1: return []
        path_indices = [u]
        while u != v:
            u = self.next_hop[u][v]
            path_indices.append(u)
        return [self.dpus[i] for i in path_indices]


class PlacementOptimizer:
    def __init__(self, dag: Dict[str, Task], network: Dict[str, DPU], links: Dict[str, Link]):
        self.dag = dag
        self.network = network
        self.links = links
        self.router = Router(network, links)
        self.compute_resources = self._get_resources_by_type('compute')
        self.storage_resources = self._get_resources_by_type('storage')
        self.avg_node_costs, self.avg_edge_costs = self._calculate_avg_costs_for_ranking()

    def _get_resources_by_type(self, res_type: str) -> List[str]:
        res_list = []
        target_names = ['arm', 'dpa'] if res_type == 'compute' else ['dram', 'ssd']
        for dpu in self.network.values():
            for r in dpu.resources.values():
                if r.name in target_names:
                    if r.type == 'compute':
                        for i in range(r.capacity): res_list.append(f"{r.id}_{i}")
                    else: res_list.append(r.id)
        return res_list

    def _calculate_avg_costs_for_ranking(self) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        storage_bws, link_bws = [], []
        for dpu in self.network.values():
            for res in dpu.resources.values():
                if res.type == 'storage' and res.bandwidth_mbps > 0: storage_bws.append(res.bandwidth_mbps)
        for link in self.links.values(): link_bws.append(link.bandwidth_gbps * 125)
        avg_storage_bw = sum(storage_bws) / len(storage_bws) if storage_bws else 1
        avg_link_bw = sum(link_bws) / len(link_bws) if link_bws else 1
        
        avg_node_costs = {}
        compute_workload = {'linear': 10, 'slice': 2, 'rope': 15, 'view': 1, 'einsum': 25, 'add': 2, 'softmax': 8}
        for task_id, task in self.dag.items():
            avg_node_costs[task_id] = compute_workload.get(task.compute_type, 5) if task.type == 'compute' else 0

        avg_edge_costs = {}
        for task_id, task in self.dag.items():
            for child_id in task.children:
                child_task = self.dag[child_id]
                cost = 0
                if task.type == 'compute' and child_task.type == 'data':
                    cost = (child_task.data_size / avg_storage_bw) * 1e6
                elif task.type == 'data' and child_task.type == 'compute':
                    read_cost = (task.data_size / avg_storage_bw) * 1e6
                    transfer_cost = (task.data_size / avg_link_bw) * 1e6
                    cost = read_cost + transfer_cost
                avg_edge_costs[(task_id, child_id)] = cost
        return avg_node_costs, avg_edge_costs

    def _compute_rank_u(self):
        for task_id in reversed(list(self.dag.keys())):
            task = self.dag[task_id]
            max_succ_rank = 0
            if task.children:
                for child_id in task.children:
                    comm_cost = self.avg_edge_costs.get((task_id, child_id), 0)
                    child_rank_u = self.dag[child_id].rank_u
                    max_succ_rank = max(max_succ_rank, comm_cost + child_rank_u)
            task.rank_u = self.avg_node_costs.get(task_id, 0) + max_succ_rank

    def _get_comm_latency(self, task_id: str, resource_id: str) -> float:
        return random.uniform(1, 5) #假定通信延迟是1-5us的随机值

    def _calculate_data_arrival_time(self, parent_id: str, child_id: str, child_res_id: str,
                                       placement: Placement, task_finish_time: Dict,
                                       link_usage: Dict[str, List[Tuple[float, float]]]) -> float:
        """这是 PlacementAlgorithm._calculate_data_arrival_time 的修复版本"""
        data_ready_time = task_finish_time.get(parent_id, 0)
        data_size = self.dag[parent_id].data_size

        source_res_id = placement[parent_id]
        
        # 建立一个资源到DPU的映射，避免每次都循环查找
        if not hasattr(self, '_res_to_dpu_map'):
            self._res_to_dpu_map = {}
            for dpu in self.network.values():
                for res_id in dpu.resources:
                    self._res_to_dpu_map[res_id] = dpu.id
                # 同样处理展开的计算核心
                for res in dpu.resources.values():
                    if res.type == 'compute':
                        for i in range(res.capacity):
                            self._res_to_dpu_map[f"{res.id}_{i}"] = dpu.id

        source_dpu_id = self._res_to_dpu_map[source_res_id]
        dest_dpu_id = self._res_to_dpu_map[child_res_id]

        if source_dpu_id == dest_dpu_id:
            # 简化：假设DPU内部通信时间可忽略不计
            return data_ready_time

        comm_path = self.router.get_path(source_dpu_id, dest_dpu_id)
        if not comm_path or len(comm_path) <= 1:
            return data_ready_time # 已经在同一个DPU或路径不存在

        # 寻找瓶颈链路
        bottleneck_bw = float('inf')
        bottleneck_link_key = ""
        for i in range(len(comm_path) - 1):
            dpu1, dpu2 = comm_path[i], comm_path[i+1]
            link_key = f"link_{dpu1}_{dpu2}"
            if link_key not in self.links: link_key = f"link_{dpu2}_{dpu1}" # 检查反向
            
            if self.links[link_key].bandwidth_gbps < bottleneck_bw:
                bottleneck_bw = self.links[link_key].bandwidth_gbps
                bottleneck_link_key = link_key

        if bottleneck_link_key == "": return data_ready_time

        # 采用简化的争用模型计算实际耗时
        N = sum(1 for start, finish in link_usage.get(bottleneck_link_key, []) if start <= data_ready_time < finish)
        B_eff = (bottleneck_bw * 1000 / 8) / (N + 1) # GB/s -> MB/s
        duration = (data_size / B_eff) * 1e6 if B_eff > 0 else float('inf') # us

        return data_ready_time + duration

    def _get_execution_time(self, task: Task, resource_id: str) -> float:
        # 确切地计算task在res_id上的执行时间
        if task.type == 'compute':
            return random.uniform(5, 20) #还未建模计算节点的执行时间，假定是个random值
        else:
            return task.data_size / self.storage_resources[resource_id].bandwidth_mbps
        
    def run_heft_fixed(self) -> Placement:
        """这是 PlacementAlgorithm.run_heft 的修复版本"""
        self._compute_rank_u()
        sorted_tasks = sorted(self.dag.values(), key=lambda t: t.rank_u, reverse=True)
        
        placement: Placement = {}
        resource_available_time = {res_id: 0 for res_id in self.compute_resources + self.storage_resources}
        link_usage: Dict[str, List[Tuple[float, float]]] = {key: [] for key in self.links.keys()}
        task_finish_time: Dict[str, float] = {}

        for task in sorted_tasks:
            target_resources = self.compute_resources if task.type == 'compute' else self.storage_resources
            best_resource = -1
            min_eft = float('inf')
            
            for res_id in target_resources:
                resource_free_time = resource_available_time[res_id]
                max_arrival_time = 0
                for parent_id in task.parents:
                    arrival_time = self._calculate_data_arrival_time(parent_id, task.id, res_id, placement, task_finish_time, link_usage)
                    max_arrival_time = max(max_arrival_time, arrival_time)
                
                est = max(resource_free_time, max_arrival_time)
                execution_time = self._get_execution_time(task, res_id)
                eft = est + execution_time
                
                if eft < min_eft:
                    min_eft = eft
                    best_resource = res_id
            
            placement[task.id] = best_resource
            task_finish_time[task.id] = min_eft
            resource_available_time[best_resource] = min_eft
            
            #预定通信资源。更新link_usage
            for parent_id in task.parents:
                comm_path = self.router.get_path(placement[parent_id], placement[task.id]) #实际上应该是两个dpu之间的路径
                start_time = task_finish_time[parent_id]
                
                B_total = float('inf')
                for i in range(len(comm_path)):
                    present_dpu = comm_path[i]
                    if i + 1 < len(comm_path):
                        next_dpu = comm_path[i + 1]
                    else: break
                    if self.links.get(f"link_{present_dpu}_{next_dpu}").bandwidth_gbps < B_total:
                        B_total = self.links.get(f"link_{present_dpu}_{next_dpu}").bandwidth_gbps
                        dpu1 = present_dpu
                        dpu2 = next_dpu
                        
                N = sum(1 for start, finish in link_usage.get(f"link_{dpu1}_{dpu2}", []) if start <= start_time < finish)
                B_eff = B_total / (N + 1)
                duration = (self.dag[parent_id].data_size / B_eff) * 1e6

                end_time = start_time + duration
                
                for i in range(len(comm_path)):
                    link_usage_key = f"link_{comm_path[i]}_{comm_path[i+1]}" if i + 1 < len(comm_path) else None
                    link_usage[link_usage_key].append((start_time, end_time)) if link_usage_key else None

        # print("HEFT finished.")
        return placement

    def run_simulated_annealing(self, initial_placement: Placement,
                                initial_temp=1000, final_temp=1, alpha=0.99, steps_per_temp=100) -> Placement:
        """
        模拟退火
        """
        # print("Running Simulated Annealing for optimization...")
        current_placement = copy.deepcopy(initial_placement)
        
        simulator = Simulator(self.dag, current_placement, self.network, self.links, self.router)
        current_cost = simulator.start_simulation()
        
        best_placement = current_placement
        best_cost = current_cost
        temp = initial_temp

        while temp > final_temp:
            for _ in range(steps_per_temp):
                new_placement = copy.deepcopy(current_placement)
                task_to_move_id = random.choice(list(self.dag.keys()))
                task_to_move = self.dag[task_to_move_id]

                if task_to_move.type == 'compute':
                    new_resource = random.choice(self.compute_resources)
                    while new_resource == new_placement[task_to_move_id]:
                        new_resource = random.choice(self.compute_resources)
                else: # 'data' task
                    new_resource = random.choice(self.storage_resources)
                    while new_resource == new_placement[task_to_move_id]:
                        new_resource = random.choice(self.storage_resources)
                new_placement[task_to_move_id] = new_resource

                simulator = Simulator(self.dag, new_placement, self.network, self.links, self.router)
                new_cost = simulator.start_simulation()
                
                cost_delta = new_cost - current_cost
                if cost_delta < 0 or random.uniform(0, 1) < math.exp(-cost_delta / temp):
                    current_placement = new_placement
                    current_cost = new_cost
                
                if current_cost < best_cost:
                    best_placement = current_placement
                    best_cost = current_cost
            
            # print(f"Temp: {temp:.2f}, Current Cost: {current_cost:.2f} us, Best Cost: {best_cost:.2f} us")
            temp *= alpha
        
        # print("Simulated Annealing finished.")
        return best_placement