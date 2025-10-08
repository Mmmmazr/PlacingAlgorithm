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
        self.avg_node_costs, self.avg_edge_costs = self._calculate_avg_costs_for_ranking()

    def _build_resource_maps(self):
        """预先计算从资源ID到DPU ID和资源对象的映射。"""
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
                avg_edge_costs[(task_id, child_id)] = (task.data_size / avg_link_bw) * 1e6
        # print(avg_node_costs, avg_edge_costs)
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
        """基于完整资源路径及其瓶颈带宽计算数据到达时间。(注意：此简化模型未考虑链路争用)"""
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
                current_bw = u_res.internal_bandwidth_mbps if u_res else float('inf')
            else:
                link_key = f"link_{u_dpu}_{v_dpu}"
                rev_link_key = f"link_{v_dpu}_{u_dpu}"
                link = self.links.get(link_key) or self.links.get(rev_link_key)
                if link:
                    current_bw = link.bandwidth_gbps * 125 
            
            bottleneck_bw_mbps = min(bottleneck_bw_mbps, current_bw)
        
        if bottleneck_bw_mbps == float('inf') or bottleneck_bw_mbps <= 0:
            return float('inf')

        duration = (data_size / bottleneck_bw_mbps) * 1e6 
        # print(f"({parent_id}, {child_res_id}): {duration}")
        return data_ready_time + duration

    def _get_execution_time(self, task: Task, resource_id: str) -> float:
        if task.type == 'compute':
            compute_workload = {'linear': 10, 'slice': 2, 'rope': 15, 'view': 1, 'einsum': 25, 'add': 2, 'softmax': 8}
            # 将来可根据resource_id的类型（如dpa vs arm）进行性能缩放
            return float(compute_workload.get(task.compute_type, 5))
        else: # 'data' task
            # 注意: 基础资源ID可能需要从展开后的ID中提取，例如'dpu0_dram_0' -> 'dpu0_dram'
            base_res_id = '_'.join(resource_id.split('_')[:-1]) if resource_id.rsplit('_', 1)[-1].isdigit() else resource_id
            storage_resource = self._res_map.get(base_res_id)
            if storage_resource and storage_resource.bandwidth_mbps > 0:
                return (task.data_size / storage_resource.bandwidth_mbps) * 1e6
            return 0.0

    def run_heft(self) -> Placement:
        # print("Running HEFT with Global Router for initial placement...")
        self._compute_rank_u()

        # print("\n--- Verifying Task Ranks (Top 5) ---")
        sorted_tasks_for_print = sorted(self.dag.values(), key=lambda t: t.rank_u, reverse=True)
        # for i in range(min(5, len(sorted_tasks_for_print))):
        #     task = sorted_tasks_for_print[i]
        #     print(f"  Rank {i+1}: Task='{task.id}' ('{task.name}'), rank_u={task.rank_u:.2f}")
        
        sorted_tasks = list(sorted_tasks_for_print)
        
        placement: Placement = {}
        # 使用“忙碌时段列表”替代单一的“可用时间点”
        resource_busy_slots: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        task_finish_time: Dict[str, float] = {}

        for task in sorted_tasks:
            target_resources = self.compute_resources if task.type == 'compute' else self.storage_resources
            best_resource = ""
            min_eft = float('inf')
            best_est = 0.0 # 记录最佳方案的开始执行时间

            # print(f"\n--- Placing Task: {task.id} ('{task.name}') ---")
            for res_id in target_resources:
                # 1. 计算数据到达时间 (Ready Time)
                ready_time = 0.0
                for parent_id in task.parents:
                    arrival_time = self._calculate_data_arrival_time(parent_id, res_id, placement, task_finish_time)
                    ready_time = max(ready_time, arrival_time)
                
                execution_time = self._get_execution_time(task, res_id)
                
                # 2. 寻找可插入的、最早的执行开始时间 (EST)
                est = 0.0
                busy_slots = sorted(resource_busy_slots.get(res_id, []))
                
                if not busy_slots:
                    est = ready_time
                else:
                    # 检查第一个空闲窗口（从0到第一个任务开始前）
                    if ready_time + execution_time <= busy_slots[0][0]:
                        est = ready_time
                    else:
                        # 遍历所有忙碌窗口之间的空隙
                        found_slot = False
                        for i in range(len(busy_slots) - 1):
                            gap_start = busy_slots[i][1]
                            gap_end = busy_slots[i+1][0]
                            
                            # 任务可以开始的最早时间是 max(数据就绪时间, 空隙开始时间)
                            potential_start = max(ready_time, gap_start)
                            
                            if potential_start + execution_time <= gap_end:
                                est = potential_start
                                found_slot = True
                                break
                        
                        # 如果所有空隙都不够，就放在最后一个任务的后面
                        if not found_slot:
                            est = max(ready_time, busy_slots[-1][1])

                current_eft = est + execution_time
                
                if current_eft < min_eft:
                    min_eft = current_eft
                    best_est = est
                    best_resource = res_id
            
            # 3. 最终确定放置并更新状态
            placement[task.id] = best_resource
            task_finish_time[task.id] = min_eft
            resource_busy_slots[best_resource].append((best_est, min_eft))

            # print(f"{best_resource}: {resource_busy_slots[best_resource]}")
            # print(f"  >> Placed on: '{best_resource}', EST={best_est:.2f}, EFT={min_eft:.2f} us")
        
        # print("\nHEFT finished.")
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

# test for router & HEFT
# if __name__ == '__main__':
#     from DpuNetwork import create_dpu_network
#     from TaskGraph import create_workflow_dag

#     print("="*40)
#     print("Running Standalone Test for PlacementAlgorithm")
#     print("="*40)

#     dag = create_workflow_dag()
#     network, links = create_dpu_network(num_dpus=4)
    
#     test_router = GlobalRouter(network, links)

#     optimizer = PlacementOptimizer(dag, network, links)
#     optimizer.router = test_router 
    
#     initial_placement = optimizer.run_heft()

#     print("\n--- HEFT Final Placement Result ---")
#     for task_id, res_id in initial_placement.items():
#         task_name = dag[task_id].name
#         dpu_id = optimizer._res_to_dpu_map.get(res_id, "Unknown DPU")
#         print(f"  - Task '{task_name}' ({task_id}) -> Resource '{res_id}' (on DPU '{dpu_id}')")

# test for HEFT
from typing import Dict
from PlacementAlgorithm import PlacementOptimizer
from BasicDefinitions import Task, DPU, Link, Resource

def create_complex_test_environment():
    """创建一个包含真实依赖、异构资源和通信成本的测试环境"""

    # 1. 创建任务图 (DAG) - Fork-Join 结构
    dag: Dict[str, Task] = {
        "T_Start": Task(id="T_Start", name="LoadData", type='data', data_size=100.0, children=["T_A", "T_C"]),
        "T_A": Task(id="T_A", name="Heavy_A", type='compute', compute_type='einsum', data_size=50.0, parents=["T_Start"], children=["T_B"]), # 产生50MB数据
        "T_C": Task(id="T_C", name="Heavy_C", type='compute', compute_type='einsum', data_size=50.0, parents=["T_Start"], children=["T_D"]), # 产生50MB数据
        "T_B": Task(id="T_B", name="Light_B", type='compute', compute_type='add', parents=["T_A"], children=["T_Finish"]),
        "T_D": Task(id="T_D", name="Light_D", type='compute', compute_type='add', parents=["T_C"], children=["T_Finish"]),
        "T_Finish": Task(id="T_Finish", name="StoreResult", type='data', data_size=1.0, parents=["T_B", "T_D"])
    }

    # 2. 创建异构网络
    network: Dict[str, DPU] = {
        "dpu0": DPU(
            id="dpu0",
            resources={
                "dpu0_dram": Resource("dpu0_dram", "dram", "storage", bandwidth_mbps=20000),
                "dpu0_dpa": Resource("dpu0_dpa", "dpa", "compute", capacity=1), # 高性能计算
                "dpu0_nic": Resource("dpu0_nic", "nic", "communication")
            },
            # DPU内部所有资源都通过NoC连接
            noc=[("dpu0_dram", "dpu0_dpa"), ("dpu0_dram", "dpu0_nic"), ("dpu0_dpa", "dpu0_nic")]
        ),
        "dpu1": DPU(
            id="dpu1",
            resources={
                "dpu1_dram": Resource("dpu1_dram", "dram", "storage", bandwidth_mbps=20000),
                "dpu1_arm": Resource("dpu1_arm", "arm", "compute", capacity=1), # 普通性能计算
                "dpu1_nic": Resource("dpu1_nic", "nic", "communication")
            },
            noc=[("dpu1_dram", "dpu1_arm"), ("dpu1_dram", "dpu1_nic"), ("dpu1_arm", "dpu1_nic")]
        )
    }

    links: Dict[str, Link] = {
        "link_dpu0_dpu1": Link("link_dpu0_dpu1", "dpu0", "dpu1", 100) # 100 Gbps link
    }
    
    return dag, network, links

class HighPerformanceOptimizer(PlacementOptimizer):
    """
    为了测试异构性，我们让DPA比ARM快5倍。
    """
    def _get_execution_time(self, task: Task, resource_id: str) -> float:
        # 调用父类的原始方法获取基础时间
        base_time = super()._get_execution_time(task, resource_id)
        if task.type == 'compute':
            if 'dpa' in resource_id:
                return base_time / 5.0  # DPA 速度是 ARM 的5倍
            else: # arm
                return base_time
        return base_time

if __name__ == '__main__':
    print("="*60)
    print(" Running Complex HEFT Verification Test ")
    print(" (Dependencies, Heterogeneous Resources, Communication Costs) ")
    print("="*60)

    # 1. 创建复杂的测试环境
    dag, network, links = create_complex_test_environment()
    
    # 2. 使用我们专为测试设计的、带性能差异的Optimizer
    optimizer = HighPerformanceOptimizer(dag, network, links)
    
    # 3. 运行真实的HEFT算法
    initial_placement = optimizer.run_heft()

    print("\n--- HEFT Final Placement Result ---")
    makespan = 0
    for task_id, res_id in initial_placement.items():
        task_name = dag[task_id].name
        dpu_id = optimizer._res_to_dpu_map.get(res_id, "Unknown DPU")
        print(f"  - Task '{task_name}' ({task_id}) -> Resource '{res_id}' (on DPU '{dpu_id}')")