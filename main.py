from PlacementAlgorithm import PlacementOptimizer
from DesSimulator import Simulator
from TaskGraph import create_workflow_dag
from DpuNetwork import create_dpu_network

def print_placement_details(title, placement, dag, optimizer):
    print(title)
    for task_id, res_id in placement.items():
        task_name = dag[task_id].name
        dpu_id = optimizer._res_to_dpu_map.get(res_id, "Unknown DPU")
        print(f"  - Task '{task_name}' ({task_id}) -> Resource '{res_id}' (on DPU '{dpu_id}')")

if __name__ == "__main__":
    print("[PHASE 1] 初始化任务图和DPU网络...")
    dag = create_workflow_dag()
    network, links = create_dpu_network(num_dpus=4)
    optimizer = PlacementOptimizer(dag, network, links)
    print("初始化完成。\n")

    print("[PHASE 2] 运行HEFT算法...")
    initial_placement = optimizer.run_heft()
    print_placement_details("\n--- HEFT Final Placement Result ---", initial_placement, dag, optimizer)
    print("\nHEFT执行完毕。\n")

    print("[PHASE 3] 使用仿真器评估HEFT方案...")
    simulator_heft = Simulator(dag, initial_placement, network, links, optimizer.router)
    heft_cost = simulator_heft.start_simulation()
    
    if heft_cost == float('inf'):
        print("HEFT基线方案因超出内存限制而无效！\n")
    else:
        print(f"HEFT基线成本(Makespan): {heft_cost:.2f} us\n")

    print("[PHASE 4] 运行模拟退火优化...")
    final_placement = optimizer.run_simulated_annealing(
        initial_placement=initial_placement,
        initial_temp=1000,
        final_temp=1,
        alpha=0.95,
        steps_per_temp=20,
        heuristic_prob=0.0
    )
    # 我试了一下，好像heuristic还没有纯随机好
    print_placement_details("\n--- Simulated Annealing Final Placement Result ---", final_placement, dag, optimizer)
    print("\n模拟退火执行完毕。\n")

    print("[PHASE 5] 仿真器评估最终方案...")
    simulator_sa = Simulator(dag, final_placement, network, links, optimizer.router)
    sa_cost = simulator_sa.start_simulation()
    
    if sa_cost == float('inf'):
         print("SA最终方案因超出内存限制而无效！\n")
    else:
        print(f"SA最终成本(Makespan): {sa_cost:.2f} us\n")

    print("="*60)
    print(" 性能对比与验证 ")
    print("="*60)
    print(f"HEFT初始成本:     {heft_cost:.2f} us")
    print(f"SA优化后成本:     {sa_cost:.2f} us")
    improvement = heft_cost - sa_cost
    improvement_percent = (improvement / heft_cost) * 100 if heft_cost > 0 else 0
    if sa_cost < heft_cost:
        print(f"\n[SUCCESS] 模拟退火找到更优解，提升 {improvement:.2f} us ({improvement_percent:.2f}%)")
    elif sa_cost == heft_cost:
        print(f"\n[INFO] 模拟退火未找到更优解，HEFT方案已近最优。")
    else:
        print(f"\n[WARNING] 模拟退火结果反而更差，建议检查实现。")