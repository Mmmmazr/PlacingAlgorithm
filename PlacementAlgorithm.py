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
            avg_node_costs[task_id] = compute_workload.get(task.compute_type, 5) if task.type == 'compute' else task.data_size /avg_storage_bw
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

    def _calculate_data_arrival_time(self, parent_id: str, child_res_id: str,
                                       placement: Placement, task_finish_time: Dict) -> float:
        """计算数据到达时间(还没考虑链路争用)"""
        data_ready_time = task_finish_time.get(parent_id, 0)
        parent_task = self.dag[parent_id]
        data_size = parent_task.data_size
        if data_size <= 1e-9:
            return data_ready_time

        source_res_id = placement[parent_id]
        if source_res_id == child_res_id:
            return data_ready_time

        comm_path = self.router.get_path(source_res_id, child_res_id)
        if not comm_path or len(comm_path) <= 1:
            return data_ready_time

        bottleneck_bw_mbps = float('inf')
        
        for i in range(len(comm_path) - 1):
            u_id, v_id = comm_path[i], comm_path[i+1]
            u_dpu = self._res_to_dpu_map.get(u_id)
            v_dpu = self._res_to_dpu_map.get(v_id)

            current_bw = float('inf')
            if u_dpu == v_dpu:
                base_u_id = '_'.join(u_id.split('_')[:-1]) if u_id.rsplit('_', 1)[-1].isdigit() else u_id
                u_res = self._res_map.get(base_u_id)
                current_bw = u_res.internal_bandwidth_MBps if u_res else float('inf')
            else:
                link_key = f"link_{u_dpu}_{v_dpu}"
                rev_link_key = f"link_{v_dpu}_{u_dpu}"
                link = self.links.get(link_key) or self.links.get(rev_link_key)
                if link:
                    current_bw = link.bandwidth_MBps
            
            bottleneck_bw_mbps = min(bottleneck_bw_mbps, current_bw)
        
        if bottleneck_bw_mbps == float('inf') or bottleneck_bw_mbps <= 0:
            return float('inf')

        duration = (data_size / bottleneck_bw_mbps) * 1e6 
        # print(f"({parent_id}, {child_res_id}): {duration}")
        return data_ready_time + duration

    def _get_execution_time(self, task: Task, resource_id: str) -> float: # 重载后可根据resource_id的类型（如dpa vs arm）进行性能调整
        if task.type == 'compute':
            compute_workload = {'linear': 10, 'slice': 2, 'rope': 15, 'view': 1, 'einsum': 25, 'add': 2, 'softmax': 8} # 单位us
            return float(compute_workload.get(task.compute_type, 5))
        else: # 'data'
            base_res_id = '_'.join(resource_id.split('_')[:-1]) if resource_id.rsplit('_', 1)[-1].isdigit() else resource_id
            storage_resource = self._res_map.get(base_res_id)
            if storage_resource and storage_resource.bandwidth_MBps > 0:
                return (task.data_size / storage_resource.bandwidth_MBps) * 1e6
            return 0.0

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
        task_finish_time: Dict[str, float] = {}
        resource_finish_times: Dict[str, List[float]] = defaultdict(list)
        for res_id in self.compute_resources + self.storage_resources:
            res_capacity = self._res_map[res_id].capacity
            resource_finish_times[res_id] = [0.0] * res_capacity

        for task in self.sorted_tasks:
            target_resources = self.compute_resources if task.type == 'compute' else self.storage_resources
            min_eft = float('inf')
            best_resource_id = ""
            best_core_idx = -1

            # print(f"\n--- 正在放置任务: {task.id} ---")
            for res_id in target_resources:
                ready_time = 0.0
                for parent_id in task.parents:
                    arrival_time = self._calculate_data_arrival_time(parent_id, res_id, placement, task_finish_time)
                    ready_time = max(ready_time, arrival_time)
                
                execution_time = self._get_execution_time(task, res_id)
                
                core_times = resource_finish_times[res_id]
                earliest_core_time = min(core_times)
                core_idx = core_times.index(earliest_core_time)

                est = max(ready_time, earliest_core_time)
                current_eft = est + execution_time
                
                # print(f"  - 尝试资源 {res_id}:")
                # print(f"    - 数据准备时间 (Ready Time): {ready_time:.2f}")
                # print(f"    - 核心可用时间 (Earliest Core Time): {earliest_core_time:.2f}")
                # print(f"    - 任务开始时间 (EST): {est:.2f}")
                # print(f"    - 执行时间: {execution_time:.2f}")
                # print(f"    - 完成时间 (EFT): {current_eft:.2f}")

                if current_eft < min_eft:
                    min_eft = current_eft
                    best_resource_id = res_id
                    best_core_idx = core_idx

            placement[task.id] = best_resource_id
            task_finish_time[task.id] = min_eft
            if best_resource_id:
                resource_finish_times[best_resource_id][best_core_idx] = min_eft

        #     print(f"  >> 决定放置在: '{best_resource_id}', 完成时间 (EFT) = {min_eft:.2f}")
        
        # print("\n--- HEFT 算法结束 ---")
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
    from TaskGraph import create_workflow_dag # 确保您的文件名和函数名正确

    print("="*40)
    print("运行 HEFT 算法测试")
    print("="*40)

    # 1. 加载 DPU 网络和任务图
    dag = create_workflow_dag(json_path=r"C:\code\PlacingAlgorithm\test_cases\TaskGraph1.json")
    network, links = create_dpu_network(json_path=r"C:\code\PlacingAlgorithm\test_cases\DpuNetwork1.json")

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