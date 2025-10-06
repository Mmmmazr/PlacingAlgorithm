import math
import random
import copy
from typing import Dict, List, Tuple
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
        self.avg_node_costs, self.avg_edge_costs = self._calculate_avg_costs_for_ranking()

    def _build_resource_maps(self):
        """预先计算从资源ID到DPU ID和资源对象的映射。"""
        self._res_to_dpu_map: Dict[str, str] = {}
        self._res_map: Dict[str, Resource] = {}
        for dpu in self.network.values():
            for res_id, res_obj in dpu.resources.items():
                self._res_map[res_id] = res_obj
                if res_obj.type == 'compute':
                    for i in range(res_obj.capacity):
                        expanded_id = f"{res_id}_{i}"
                        self._res_to_dpu_map[expanded_id] = dpu.id
                else:
                    self._res_to_dpu_map[res_id] = dpu.id

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

    def _calculate_avg_costs_for_ranking(self):
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
                if task.type == 'compute' and child_task.type == 'data': cost = (child_task.data_size / avg_storage_bw) * 1e6
                elif task.type == 'data' and child_task.type == 'compute': cost = ((task.data_size / avg_storage_bw) + (task.data_size / avg_link_bw)) * 1e6
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

    def _calculate_data_arrival_time(self, parent_id: str, child_res_id: str,
                                       placement: Placement, task_finish_time: Dict) -> float:
        """基于完整资源路径及其瓶颈带宽计算数据到达时间。"""
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
        
        # 沿整条路径寻找瓶颈带宽
        for i in range(len(comm_path) - 1):
            u_id, v_id = comm_path[i], comm_path[i+1]
            u_dpu = self._res_to_dpu_map.get(u_id)
            v_dpu = self._res_to_dpu_map.get(v_id)

            current_bw = float('inf')
            if u_dpu == v_dpu: # DPU内部传输 (NoC)
                # 从Resource定义中简化NoC带宽估算
                # 注意：这里我们无法轻易地区分基础资源ID和展开后的ID，所以我们用startwith来查找
                base_u_id = '_'.join(u_id.split('_')[:-1]) if u_id.rsplit('_', 1)[-1].isdigit() else u_id
                u_res = self._res_map.get(base_u_id)
                current_bw = u_res.internal_bandwidth_mbps if u_res else float('inf')
            else: # DPU之间传输 (Link)
                link_key = f"link_{u_dpu}_{v_dpu}"
                rev_link_key = f"link_{v_dpu}_{u_dpu}"
                link = self.links.get(link_key) or self.links.get(rev_link_key)
                if link:
                    current_bw = link.bandwidth_gbps * 1000 / 8 # 转换为 MB/s
            
            bottleneck_bw_mbps = min(bottleneck_bw_mbps, current_bw)
        
        if bottleneck_bw_mbps == float('inf') or bottleneck_bw_mbps <= 0:
            return float('inf')

        # HEFT的粗略估算忽略争用
        duration = (data_size / bottleneck_bw_mbps) * 1e6 # in microseconds
        return data_ready_time + duration

    def _get_execution_time(self, task: Task, resource_id: str) -> float:
        if task.type == 'compute':
            return random.uniform(5, 20) 
        else: # 'data' task
            storage_resource = self._res_map.get(resource_id)
            if storage_resource and storage_resource.bandwidth_mbps > 0:
                return (task.data_size / storage_resource.bandwidth_mbps) * 1e6
            return 0.0

    def run_heft(self) -> Placement:
        print("Running HEFT with Global Router for initial placement...")
        self._compute_rank_u()

        # --- Verification Statement ---
        print("\n--- Verifying Task Ranks (Top 5) ---")
        sorted_tasks_for_print = sorted(self.dag.values(), key=lambda t: t.rank_u, reverse=True)
        for i in range(min(5, len(sorted_tasks_for_print))):
            task = sorted_tasks_for_print[i]
            print(f"  Rank {i+1}: Task='{task.id}' ('{task.name}'), rank_u={task.rank_u:.2f}")
        
        sorted_tasks = list(sorted_tasks_for_print)
        
        placement: Placement = {}
        resource_available_time = {res_id: 0 for res_id in self.compute_resources + self.storage_resources}
        task_finish_time: Dict[str, float] = {}

        for task in sorted_tasks:
            target_resources = self.compute_resources if task.type == 'compute' else self.storage_resources
            best_resource = ""
            min_eft = float('inf')
            
            # --- Verification Statement ---
            print(f"\n--- Placing Task: {task.id} ('{task.name}') ---")
            eft_samples = {}
            # 仅为前几个候选资源打印示例EFT以避免刷屏
            for i in range(min(3, len(target_resources))):
                res_id = random.choice(target_resources) # 随机采样几个
                resource_free_time = resource_available_time[res_id]
                max_arrival_time = 0
                for parent_id in task.parents:
                    arrival_time = self._calculate_data_arrival_time(parent_id, res_id, placement, task_finish_time)
                    max_arrival_time = max(max_arrival_time, arrival_time)
                est = max(resource_free_time, max_arrival_time)
                execution_time = self._get_execution_time(task, res_id)
                eft = est + execution_time
                eft_samples[res_id] = f"{eft:.2f} us"
            print(f"  Sample EFTs: {eft_samples}")
            
            for res_id in target_resources:
                resource_free_time = resource_available_time[res_id]
                max_arrival_time = 0
                for parent_id in task.parents:
                    arrival_time = self._calculate_data_arrival_time(parent_id, res_id, placement, task_finish_time)
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

            # --- Verification Statement ---
            print(f"  >> Placed on: '{best_resource}' with final EFT: {min_eft:.2f} us")
        
        print("\nHEFT finished.")
        return placement

    def run_simulated_annealing(self, initial_placement: Placement,
                                initial_temp=1000, final_temp=1, alpha=0.99, steps_per_temp=100) -> Placement:
        print("Running Simulated Annealing for optimization...")
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
            
            print(f"Temp: {temp:.2f}, Current Cost: {current_cost:.2f} us, Best Cost: {best_cost:.2f} us")
            temp *= alpha
        
        print("Simulated Annealing finished.")
        return best_placement

# --- Standalone Test Block ---
if __name__ == '__main__':
    # 导入所需的创建函数
    from DpuNetwork import create_dpu_network
    from TaskGraph import create_workflow_dag

    print("="*40)
    print("Running Standalone Test for PlacementAlgorithm")
    print("="*40)

    # 1. 创建模拟环境
    dag = create_workflow_dag()
    network, links = create_dpu_network(num_dpus=4)
    
    # 2. 初始化并验证 GlobalRouter
    test_router = GlobalRouter(network, links)
    test_router.print_sample_paths()

    # 3. 初始化 PlacementOptimizer 并运行 HEFT
    # optimizer = PlacementOptimizer(dag, network, links)
    
    # # 在优化器上设置相同的测试路由器实例以避免重复计算
    # optimizer.router = test_router 
    
    # initial_placement = optimizer.run_heft()

    # print("\n--- HEFT Final Placement Result ---")
    # for task_id, res_id in initial_placement.items():
    #     print(f"  - Task '{dag[task_id].name}' ({task_id}) -> Resource '{res_id}'")