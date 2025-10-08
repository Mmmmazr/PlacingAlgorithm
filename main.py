import time
from typing import Dict

# 从您提供的项目文件中导入必要的模块和函数
from PlacementAlgorithm import PlacementOptimizer
from DesSimulator import Simulator
from BasicDefinitions import Task, DPU, Link, Resource, Placement
from DpuNetwork import create_dpu_network

# --- 我们将您提供的测试用例代码直接整合到这个main.py中 ---

def create_complex_test_environment():
    """创建一个包含真实依赖、异构资源和通信成本的测试环境"""
    # 1. 创建任务图 (DAG) - Fork-Join 结构
    dag: Dict[str, Task] = {
        "T_Start": Task(id="T_Start", name="LoadData", type='data', data_size=100.0, children=["T_A", "T_C"]),
        "T_A": Task(id="T_A", name="Heavy_A", type='compute', compute_type='einsum', data_size=50.0, parents=["T_Start"], children=["T_B"]),
        "T_C": Task(id="T_C", name="Heavy_C", type='compute', compute_type='einsum', data_size=50.0, parents=["T_Start"], children=["T_D"]),
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
    """为了测试异构性，我们让DPA比ARM快5倍。"""
    def _get_execution_time(self, task: Task, resource_id: str) -> float:
        base_time = super()._get_execution_time(task, resource_id)
        if task.type == 'compute':
            if 'dpa' in resource_id:
                return base_time / 5.0
            else: # arm
                return base_time
        return base_time

def print_placement_details(title: str, placement: Placement, dag: Dict[str, Task], optimizer: PlacementOptimizer):
    """一个辅助函数，用于格式化打印放置方案。"""
    print(title)
    for task_id, res_id in placement.items():
        task_name = dag[task_id].name
        dpu_id = optimizer._res_to_dpu_map.get(res_id, "Unknown DPU")
        print(f"  - Task '{task_name}' ({task_id}) -> Resource '{res_id}' (on DPU '{dpu_id}')")

if __name__ == '__main__':
    start_time = time.time()
    
    # ==================================================================
    # 步骤 1: 初始化环境和优化器
    # ==================================================================
    print("[PHASE 1] Initializing Environment and Optimizer...")
    dag, network, links = create_complex_test_environment()
    optimizer = HighPerformanceOptimizer(dag, network, links)
    print("Initialization complete.\n")

    # ==================================================================
    # 步骤 2: 运行 HEFT 获取高质量的初始解
    # ==================================================================
    print("[PHASE 2] Running HEFT to get an initial placement...")
    initial_placement = optimizer.run_heft()
    print_placement_details("\n--- HEFT Final Placement Result ---", initial_placement, dag, optimizer)
    print("\nHEFT execution finished.\n")

    # ==================================================================
    # 步骤 3: 使用仿真器评估 HEFT 的性能 (获取基准成本)
    # ==================================================================
    print("[PHASE 3] Evaluating HEFT placement with the DES Simulator...")
    # 注意：每次评估都需要一个新的Simulator实例，因为SimPy环境不能重复运行
    simulator_heft = Simulator(dag, initial_placement, network, links, optimizer.router)
    heft_cost = simulator_heft.start_simulation()
    print(f"Evaluation complete. HEFT Baseline Cost (Makespan): {heft_cost:.2f} us\n")

    # ==================================================================
    # 步骤 4: 运行模拟退火进行优化
    # ==================================================================
    print("[PHASE 4] Running Simulated Annealing for optimization...")
    # 为了快速测试，可以减少参数，例如: steps_per_temp=20, alpha=0.95
    # 使用默认参数以获得更好的优化效果
    final_placement = optimizer.run_simulated_annealing(
        initial_placement=initial_placement,
        initial_temp=1000,
        final_temp=1,
        alpha=0.95,
        steps_per_temp=20 # 在测试中适当减小以加快速度
    )
    print_placement_details("\n--- Simulated Annealing Final Placement Result ---", final_placement, dag, optimizer)
    print("\nSimulated Annealing execution finished.\n")

    # ==================================================================
    # 步骤 5: 使用仿真器评估模拟退火的最终性能
    # ==================================================================
    print("[PHASE 5] Evaluating the final placement with the DES Simulator...")
    simulator_sa = Simulator(dag, final_placement, network, links, optimizer.router)
    sa_cost = simulator_sa.start_simulation()
    print(f"Evaluation complete. SA Final Cost (Makespan): {sa_cost:.2f} us\n")

    # ==================================================================
    # 步骤 6: 测试与验证 - 对比结果
    # ==================================================================
    print("="*60)
    print(" Final Performance Comparison & Verification ")
    print("="*60)
    print(f"Initial Cost (from HEFT):     {heft_cost:.2f} us")
    print(f"Optimized Cost (from SA):     {sa_cost:.2f} us")
    
    improvement = heft_cost - sa_cost
    improvement_percent = (improvement / heft_cost) * 100 if heft_cost > 0 else 0

    if sa_cost < heft_cost:
        print(f"\n[SUCCESS] Simulated Annealing found a better solution!")
        print(f"          Improvement of {improvement:.2f} us ({improvement_percent:.2f}%)")
    elif sa_cost == heft_cost:
        print(f"\n[INFO] Simulated Annealing did not find a better solution.")
        print(f"         The initial HEFT solution was likely already optimal or near-optimal.")
    else:
        # 这种情况理论上不应发生，因为SA会保留历史最优解
        print(f"\n[WARNING] Simulated Annealing resulted in a worse solution.")
        print(f"           This might indicate an issue in the SA implementation's tracking of the best state.")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")