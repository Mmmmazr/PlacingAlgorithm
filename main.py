import copy
import time
from typing import Dict, List, Tuple

# 从你提供的文件中导入所有必要的类
from BasicDefinitions import Task, Placement, DPU, Link, Resource
from TaskGraph import create_workflow_dag
from DpuNetwork import create_dpu_network 
from PlacementAlgorithm import PlacementOptimizer, Router
from DesSimulator import Simulator

def evaluate_placement(dag, placement, network, links, router) -> float:
    """
    一个辅助函数，用于调用DES来评估任何给定放置方案的成本（总执行时间）。
    """
    simulator = Simulator(dag, placement, network, links, router)
    makespan = simulator.start_simulation()
    return makespan

if __name__ == "__main__":
    
    # --- 1. 初始化工作流和硬件环境 ---
    print("="*30)
    print("Step 1: Initializing DAG and DPU Network")
    print("="*30)
    dag = create_workflow_dag()
    network, links = create_dpu_network(num_dpus=4)
    router = Router(network, links)
    
    # --- 2. 使用 HEFT 生成初始放置方案 ---
    print("\n" + "="*30)
    print("Step 2: Running HEFT to get initial placement")
    print("="*30)
    
    optimizer = PlacementOptimizer(dag, network, links)
    
    start_time = time.time()
    initial_placement = optimizer.run_heft()
    heft_time = time.time() - start_time

    print(f"\nHEFT execution time: {heft_time:.4f} seconds.")
    print("Initial Placement by HEFT:")
    for task_id, res_id in initial_placement.items():
        print(f"  - Task '{dag[task_id].name}' ({task_id}) -> Resource '{res_id}'")

    # --- 3. 评估 HEFT 方案的真实性能 ---
    print("\n" + "="*30)
    print("Step 3: Evaluating HEFT placement with DES")
    print("="*30)
    initial_cost = evaluate_placement(copy.deepcopy(dag), initial_placement, network, links, router)
    print(f"DES evaluated cost for HEFT placement: {initial_cost:.2f} us")
    
    # --- 4. 使用模拟退火进行优化 ---
    print("\n" + "="*30)
    print("Step 4: Running Simulated Annealing for optimization")
    print("="*30)
    
    start_time = time.time()
    # 参数可以根据问题的复杂度和需要调整
    best_placement = optimizer.run_simulated_annealing(
        initial_placement=initial_placement,
        initial_temp=1000,
        final_temp=1,
        alpha=0.99,
        steps_per_temp=50  # 减少步数以加快示例运行
    )
    sa_time = time.time() - start_time
    
    print(f"\nSimulated Annealing execution time: {sa_time:.4f} seconds.")

    # --- 5. 评估最终方案并报告结果 ---
    print("\n" + "="*30)
    print("Step 5: Final Evaluation and Report")
    print("="*30)
    final_cost = evaluate_placement(copy.deepcopy(dag), best_placement, network, links, router)

    print("Final Optimized Placement by SA:")
    for task_id, res_id in best_placement.items():
        print(f"  - Task '{dag[task_id].name}' ({task_id}) -> Resource '{res_id}'")

    print("\n--- Performance Summary ---")
    print(f"Initial Cost (from HEFT): {initial_cost:.2f} us")
    print(f"Final Cost (after SA):    {final_cost:.2f} us")
    improvement = ((initial_cost - final_cost) / initial_cost) * 100 if initial_cost > 0 else 0
    print(f"Improvement: {improvement:.2f}%")