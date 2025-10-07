# test_heft_insertion.py
from typing import Dict, List, Tuple
from collections import defaultdict
from PlacementAlgorithm import PlacementOptimizer
from BasicDefinitions import Task, Placement, DPU, Link, Resource
from DesSimulator import GlobalRouter # 假设这些导入都有效

def create_test_environment():
    """创建一个专门用于测试插入逻辑的简单环境"""
    
    # 1. 创建任务图 (DAG)
    dag: Dict[str, Task] = {
        "T_A": Task(id="T_A", name="Task_A_HighPrio", type='compute', compute_type='linear'),
        "T_B": Task(id="T_B", name="Task_B_LowPrio", type='compute', compute_type='add'),
        "T_C": Task(id="T_C", name="Task_C_MidPrio", type='compute', compute_type='einsum'),
    }
    # 手动设置rank_u来控制调度顺序: A -> C -> B
    dag["T_A"].rank_u = 300
    dag["T_C"].rank_u = 200
    dag["T_B"].rank_u = 100
    
    # 2. 创建网络
    network: Dict[str, DPU] = {
        "dpu0": DPU(
            id="dpu0",
            resources={"dpu0_arm": Resource("dpu0_arm", "arm", "compute", capacity=1)}
        )
    }
    links: Dict[str, Link] = {}
    
    return dag, network, links

class MockPlacementOptimizer(PlacementOptimizer):
    """
    一个用于测试的子类，重载了部分方法以精确控制测试条件。
    """
    def __init__(self, dag, network, links):
        super().__init__(dag, network, links)
        self.currently_placing_task_id = None # 初始化属性

    def _get_execution_time(self, task: Task, resource_id: str) -> float:
        """为测试任务返回固定的执行时间"""
        if task.id == "T_A": return 30.0
        if task.id == "T_B": return 20.0
        if task.id == "T_C": return 25.0
        return 0.0

    def _calculate_data_arrival_time(self, parent_id: str, child_res_id: str,
                                       placement: Placement, task_finish_time: Dict) -> float:
        """为Task C伪造一个很晚的数据到达时间"""
        if self.currently_placing_task_id == 'T_C':
            return 80.0 # 关键条件：T_C在t=80时才就绪
        return 0.0 # 其他任务都在t=0时就绪

    def run_heft(self) -> Placement:
        print("Running MOCKED HEFT for insertion test...")
        sorted_tasks = sorted(self.dag.values(), key=lambda t: t.rank_u, reverse=True)
        
        placement: Dict[str, str] = {}
        task_finish_time: Dict[str, float] = {}
        # 【已修正】在此处添加了对 resource_busy_slots 的初始化
        resource_busy_slots: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for task in sorted_tasks:
            self.currently_placing_task_id = task.id
            target_resources = self.compute_resources
            best_resource = ""
            min_eft = float('inf')
            best_est = 0.0

            print(f"\n--- Placing Task: {task.id} ('{task.name}') ---")
            
            for res_id in target_resources:
                ready_time = self._calculate_data_arrival_time(None, res_id, placement, task_finish_time)
                execution_time = self._get_execution_time(task, res_id)
                
                est = 0.0
                busy_slots = sorted(resource_busy_slots.get(res_id, []))
                
                if not busy_slots:
                    est = ready_time
                else:
                    if ready_time + execution_time <= busy_slots[0][0]:
                        est = ready_time
                    else:
                        found_slot = False
                        for i in range(len(busy_slots) - 1):
                            gap_start = busy_slots[i][1]
                            gap_end = busy_slots[i+1][0]
                            potential_start = max(ready_time, gap_start)
                            if potential_start + execution_time <= gap_end:
                                est = potential_start
                                found_slot = True
                                break
                        if not found_slot:
                            est = max(ready_time, busy_slots[-1][1])

                current_eft = est + execution_time
                
                if current_eft < min_eft:
                    min_eft = current_eft
                    best_est = est
                    best_resource = res_id
            
            placement[task.id] = best_resource
            task_finish_time[task.id] = min_eft
            resource_busy_slots[best_resource].append((best_est, min_eft))
            
            final_slots = sorted(resource_busy_slots[best_resource])
            print(f"  >> Placed on: '{best_resource}' with DataReady={ready_time:.2f}, EST={best_est:.2f}, EFT={min_eft:.2f} us")
            print(f"  >> Resource '{best_resource}' busy slots: {final_slots}")
        
        print("\nHEFT test finished.")
        return placement

if __name__ == '__main__':
    print("="*50)
    print("Running HEFT Insertion Logic Verification Test")
    print("="*50)

    dag, network, links = create_test_environment()
    
    # 为了让测试脚本独立运行，我们需要确保它能找到 PlacementAlgorithm
    # 这里我们直接在脚本内定义了 MockPlacementOptimizer，所以没有问题
    mock_optimizer = MockPlacementOptimizer(dag, network, links)
    final_placement = mock_optimizer.run_heft()
    
    print("\n--- Final Placement Result ---")
    for task_id, res_id in final_placement.items():
        print(f"  - Task '{task_id}' -> Resource '{res_id}'")