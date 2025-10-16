import math
import random
import copy
from typing import Dict, List, Tuple
from collections import defaultdict
from BasicDefinitions import Task, Placement, DPU, Link, Resource
from DesSimulator import Simulator, GlobalRouter

class PlacementOptimizer:
    def __init__(self, dag: Dict[str, Task], network: Dict[str, DPU], links: Dict[str, Link]):
        self.dag = dag
        self.network = network
        self.links = links
        self.router = GlobalRouter(network, links)
        self.compute_resources = self._get_resources_by_type('compute')
        self.storage_resources = self._get_resources_by_type('storage')
        self._build_resource_maps()
        self.avg_node_costs, self.avg_edge_costs = self._calculate_avg_costs()
        # 预处理rank_u
        self._compute_rank_u()
        self.task_ranks = {task_id: task.rank_u for task_id, task in self.dag.items()}
        self.sorted_tasks = sorted(self.dag.values(), key=lambda task: self.task_ranks[task.id], reverse=True)

    def _build_resource_maps(self):
        """计算从资源ID到DPU ID和资源对象的映射"""
        self._res_to_dpu_map: Dict[str, str] = {}
        self._res_map: Dict[str, Resource] = {}
        for dpu in self.network.values():
            for res_id, res_obj in dpu.resources.items():
                self._res_map[res_id] = res_obj
                self._res_to_dpu_map[res_id] = dpu.id

    def _get_resources_by_type(self, res_type: str) -> List[str]:
        res_list = []
        target_names = ['arm', 'dpa'] if res_type == 'compute' else ['dram', 'ssd']
        for dpu in self.network.values():
            for r in dpu.resources.values():
                if r.name in target_names:
                    res_list.append(r.id)
        return res_list

    def _calculate_avg_costs(self):
        storage_bws, link_bws = [], []
        for dpu in self.network.values():
            for res in dpu.resources.values():
                if res.type == 'storage' and res.bandwidth_MBps > 0: 
                    storage_bws.append(res.bandwidth_MBps)
        for link in self.links.values(): 
            link_bws.append(link.bandwidth_MBps)
        avg_storage_bw = sum(storage_bws) / len(storage_bws) if storage_bws else 1
        avg_link_bw = sum(link_bws) / len(link_bws) if link_bws else 1
        avg_node_costs = {}
        compute_workload = {'linear': 10, 'slice': 2, 'rope': 15, 'view': 1, 'einsum': 25, 'add': 2, 'softmax': 8}
        for task_id, task in self.dag.items():
            avg_node_costs[task_id] = compute_workload.get(task.compute_type, 5) if task.type == 'compute' else task.data_size / avg_storage_bw * 1e6
        avg_edge_costs = {}
        for task_id, task in self.dag.items():
            for child_id in task.children:
                avg_edge_costs[(task_id, child_id)] = (task.data_size / avg_link_bw) * 1e6
        
        # print(f"平均节点成本: {avg_node_costs}")
        # print(f"平均边成本: {avg_edge_costs}")

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

    def _get_arrival_times(self, task_id: str, dest_res_id: str, placement: Placement, du_finish_times: Dict[str, List[float]]) -> List[float]:
        task = self.dag[task_id]
        du_num = task.du_num if task.du_num > 0 else 1
        du_arrival_times = [0.0] * du_num

        for parent_id in task.parents:
            parent_task = self.dag[parent_id]
            parent_du_num = parent_task.du_num if parent_task.du_num > 0 else 1
            parent_du_size = parent_task.du_size if parent_task.du_size > 0 else parent_task.data_size / parent_du_num
            
            source_res_id = placement[parent_id]
            comm_time_du = 0
            if source_res_id != dest_res_id and parent_du_size > 0:
                path = self.router.get_path(source_res_id, dest_res_id)
                bottleneck_bw = float('inf')
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    u_dpu, v_dpu = self._res_to_dpu_map[u], self._res_to_dpu_map[v]
                    current_bw = float('inf')
                    if u_dpu == v_dpu:
                        current_bw = self._res_map[u].internal_bandwidth_MBps
                    else:
                        link = self.links.get(f"link_{u_dpu}_{v_dpu}") or self.links.get(f"link_{v_dpu}_{u_dpu}")
                        if link: 
                            current_bw = link.bandwidth_MBps
                    bottleneck_bw = min(bottleneck_bw, current_bw)
                if bottleneck_bw != float('inf'):
                    comm_time_du = (parent_du_size / bottleneck_bw) * 1e6
            
            parent_finish_times = du_finish_times[parent_id]
            if parent_du_num == du_num: # 流水线模式
                for i in range(du_num):
                    arrival = parent_finish_times[i] + comm_time_du
                    du_arrival_times[i] = max(du_arrival_times[i], arrival)
            else: 
                last_parent_du_finish = parent_finish_times[-1]
                arrival = last_parent_du_finish + comm_time_du
                for i in range(du_num):
                    du_arrival_times[i] = max(du_arrival_times[i], arrival)

        return du_arrival_times

    def _get_execution_time(self, task: Task, resource_id: str) -> float: # 重载后可根据resource_id的类型（如dpa vs arm）进行性能调整
        """计算单个DU的执行时间"""
        du_num = task.du_num if task.du_num > 0 else 1
        if task.type == 'compute':
            compute_workload = {'linear': 10, 'slice': 2, 'rope': 15, 'view': 1, 'einsum': 25, 'add': 2, 'softmax': 8}
            total_workload = float(compute_workload.get(task.compute_type, 5))
            return total_workload / du_num
        else: # 'data'
            storage_resource = self._res_map.get(resource_id)
            if storage_resource and storage_resource.bandwidth_MBps > 0 and task.du_size > 0:
                return (task.du_size / storage_resource.bandwidth_MBps) * 1e6
            return 0.0

    def _find_slots(self, core_busy_slots: List[Tuple[float, float]], du_arrival_times: List[float], du_exec_time: float) -> Tuple[float, float]:
        du_num = len(du_arrival_times)
        start = du_arrival_times[0]
        
        while True:
            current_start = start
            is_valid_slot = True
            
            du_start_times = [0.0] * du_num
            du_start_times[0] = current_start
            
            for i in range(1, du_num):
                prev_du_finish = du_start_times[i-1] + du_exec_time
                du_start_times[i] = max(prev_du_finish, du_arrival_times[i])

            start_time = du_start_times[0]
            end_time = du_start_times[-1] + du_exec_time
            
            for busy_start, busy_end in core_busy_slots:
                if start_time < busy_end and end_time > busy_start:
                    is_valid_slot = False
                    start = busy_end
                    break
            
            if is_valid_slot:
                return start_time, end_time

    def _get_peak_memory_usage(self, existing_intervals: List[Tuple[float, float, float]],
                                 start_time: float, end_time: float) -> float:
        """扫描线"""
        if start_time >= end_time:
            return 0.0

        events = []
        for l, r, sz in existing_intervals:
            events.append((l, sz))  
            events.append((r, -sz)) 
        events.sort()

        peak_usage = 0.0
        current_usage = 0.0
        event_idx = 0
        while event_idx < len(events) and events[event_idx][0] < start_time:
            current_usage += events[event_idx][1]
            event_idx += 1
        peak_usage = current_usage

        while event_idx < len(events) and events[event_idx][0] <= end_time:
            current_usage += events[event_idx][1]
            peak_usage = max(peak_usage, current_usage)
            event_idx += 1
            
        return peak_usage

    def run_heft(self) -> Placement:
        # print("\n--- HEFT 算法开始运行 ---")
        
        # sorted_tasks = self.sorted_tasks
        # print("\n--- 任务排序 (基于 rank_u 降序) ---")
        # for task in sorted_tasks:
        #     print(f"  Task: {task.id}, Rank: {task.rank_u:.2f}")
            
        placement: Placement = {}
        du_finish_times: Dict[str, List[float]] = defaultdict(list)
        resource_busy_slots: Dict[str, List[List[Tuple[float, float]]]] = defaultdict(lambda: [[] for _ in range(500)]) # 假设最多100核
        storage_occupancy: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

        for task in self.sorted_tasks:
            # print(f"\n\n{'='*20} 正在放置任务: {task.id} (Type: {task.type}, DUs: {task.du_num}) {'='*20}")

            target_resources = self.compute_resources if task.type == 'compute' else self.storage_resources
            best_choice = {
                "eft": float('inf'),
                "res_id": None,
                "core_idx": -1,
                "start_time": -1.0,
                "final_du_finish_times": []
            }

            for res_id in target_resources:
                if task.type == "data" and len(task.parents) == 0:
                    du_arrival_times = [i * getattr(task, 'du_interval', 10) for i in range(task.du_num)]
                else: du_arrival_times = self._get_arrival_times(task.id, res_id, placement, du_finish_times)
                
                # print(f"     - DU 到达时间 (Ready Times): {[f'{t:.2f}' for t in du_arrival_times]}")
                du_exec_time = self._get_execution_time(task, res_id)
                # print(f"     - 单个 DU 执行时间: {du_exec_time:.2f} us")

                res_obj = self._res_map[res_id]
                for core_idx in range(res_obj.capacity):
                    # print(f"    - 核心 #{core_idx}:")
                    core_slots = resource_busy_slots[res_id][core_idx]
                    est, eft = self._find_slots(core_slots, du_arrival_times, du_exec_time)
                    # print(f"      - 当前忙碌时段: {[(f'[{s:.2f}, {e:.2f}]') for s, e in core_slots]}")
                    # print(f"      - 找到的最佳时间槽: EST = {est:.2f}, EFT = {eft:.2f}")

                    if task.type == 'data':
                        required_mem_gb = task.data_size / 1024.0
                        if required_mem_gb > res_obj.memory:
                            continue 
                        peak_usage_mb = self._get_peak_memory_usage(storage_occupancy[res_id], est, eft)
                        if (peak_usage_mb + task.data_size) > res_obj.memory * 1024:
                            continue 

                    if eft < best_choice["eft"]:
                        # print(f"      - 新的最优选择: EFT {eft:.2f} < 当前最优 {best_choice['eft']:.2f}. 更新选择。")
                        best_choice["eft"] = eft
                        best_choice["res_id"] = res_id
                        best_choice["core_idx"] = core_idx
                        best_choice["start_time"] = est
                        final_du_starts = [0.0] * len(du_arrival_times)
                        final_du_starts[0] = est
                        for i in range(1, len(du_arrival_times)):
                            prev_finish = final_du_starts[i-1] + du_exec_time
                            final_du_starts[i] = max(prev_finish, du_arrival_times[i])
                        best_choice["final_du_finish_times"] = [s + du_exec_time for s in final_du_starts]
            
            # 更新
            best_res = best_choice["res_id"]
            best_core = best_choice["core_idx"]

            # print(f"\n  [DECISION] 任务 '{task.id}' 最终决定放置在: [{best_res}] 的核心 #[{best_core}]")
            # print(f"    - 任务流开始时间 (EST): {best_choice['start_time']:.2f}")
            # print(f"    - 任务流结束时间 (EFT): {best_choice['eft']:.2f}")
            # print(f"    - 各 DU 完成时间: {[f'{t:.2f}' for t in best_choice['final_du_finish_times']]}")

            placement[task.id] = best_res
            du_finish_times[task.id] = best_choice["final_du_finish_times"]
            new_busy_slot = (best_choice["start_time"], best_choice["eft"])
            resource_busy_slots[best_res][best_core].append(new_busy_slot)
            resource_busy_slots[best_res][best_core].sort()
            if task.type == 'data':
                storage_occupancy[best_res].append((best_choice["start_time"], best_choice["eft"], task.data_size))
                # print(f"    - 更新 [{best_res}] 的存储占用.")

        return placement

    def _random_method(self, placement: Placement) -> Placement:
        '''旧方案，纯随机'''
        new_placement = copy.deepcopy(placement)
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
        return new_placement
        
    def _heuristic_method(self, placement: Placement) -> Placement:
        '''启发式，离自己最大的父亲越近越好'''
        new_placement = copy.deepcopy(placement)
        ranks = [self.task_ranks[task.id] for task in self.sorted_tasks]
        total_rank = sum(ranks)
        weights = [rank / total_rank + 1e-6 for rank in ranks]
        task_to_move_id = random.choices(self.sorted_tasks, k=1, weights=weights)[0].id
        task_to_move = self.dag[task_to_move_id]

        important_parent = None
        max_data_size = -float('inf')
        for parent_id in task_to_move.parents:
            parent_task = self.dag[parent_id]
            if parent_task.data_size > max_data_size:
                max_data_size = parent_task.data_size
                important_parent = parent_id
        
        if important_parent is not None:
            parent_resource = placement[parent_id]
            target_dpu = self._res_to_dpu_map.get(parent_resource)

            resource_choices = []
            resource_type = task_to_move.type
            for res_id in self.network[target_dpu].resources:
                if self._res_map[res_id].type == resource_type:
                    resource_choices.append(res_id)
            
            if resource_choices:
                new_resource = random.choice(resource_choices)
                new_placement[task_to_move_id] = new_resource
                return new_placement
        
        return self._random_method(placement)

    # 模拟退火
    def run_simulated_annealing(self, initial_placement: Placement,
                                initial_temp=1000, final_temp=1, alpha=0.99, 
                                steps_per_temp=100, heuristic_prob=0.7) -> Placement:
        # heuristic_prob为random_method留有一定余地
        current_placement = copy.deepcopy(initial_placement)
        
        simulator = Simulator(self.dag, current_placement, self.network, self.links, self.router)
        current_cost = simulator.start_simulation()
        
        best_placement = current_placement
        best_cost = current_cost
        temp = initial_temp

        while temp > final_temp:
            for _ in range(steps_per_temp):
                if random.random() < heuristic_prob:
                    new_placement = self._heuristic_method(current_placement)
                else:
                    new_placement = self._random_method(current_placement)

                
                simulator = Simulator(self.dag, new_placement, self.network, self.links, self.router)
                new_cost = simulator.start_simulation()
                
                cost_delta = new_cost - current_cost
                if cost_delta < 0 or random.uniform(0, 1) < math.exp(-cost_delta / temp):
                    current_placement = new_placement
                    current_cost = new_cost
                
                if current_cost < best_cost:
                    best_placement = current_placement
                    best_cost = current_cost
            
            print(f"Temp: {temp:.2f}, Current Cost: {current_cost:.2f} us, Best Cost: {best_cost:.2f} us")
            temp *= alpha
        
        return best_placement
    

if __name__ == '__main__':
    from DpuNetwork import create_dpu_network
    from TaskGraph import create_workflow_dag 

    print("="*40)
    print("运行 HEFT 算法测试")
    print("="*40)

    # 1. 加载 DPU 网络和任务图
    dag = create_workflow_dag(json_path=r"C:\code\PlacingAlgorithm\test_cases\TaskGraph1.json")
    network, links = create_dpu_network(json_path=r"C:\code\PlacingAlgorithm\test_cases\DpuNetwork1.json")
    setattr(dag['T_A'], 'du_interval', 10.0) 

    # 2. 初始化优化器
    optimizer = PlacementOptimizer(dag, network, links)
    
    # 3. 运行 HEFT 算法
    final_placement = optimizer.run_heft()

    # 4. 打印最终放置结果
    print("\n--- HEFT 最终放置结果 ---")
    for task_id, res_id in final_placement.items():
        task_name = dag[task_id].name
        dpu_id = optimizer._res_to_dpu_map.get(res_id, "未知 DPU")
        print(f"  - 任务 '{task_name}' ({task_id}) -> 资源 '{res_id}' (在 DPU '{dpu_id}' 上)")