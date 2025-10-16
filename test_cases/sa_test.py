from PlacementAlgorithm import PlacementOptimizer
from DesSimulator import Simulator
from TaskGraph import create_workflow_dag
from DpuNetwork import create_dpu_network
import random

def generate_random_placement(dag, optimizer):
        placement = {}
        compute_choices = optimizer.compute_resources
        storage_choices = optimizer.storage_resources
        for task_id, task in dag.items():
            if task.type == 'compute':
                placement[task_id] = random.choice(compute_choices)
            else:
                placement[task_id] = random.choice(storage_choices)
        return placement

if __name__ == "__main__":

    dag = create_workflow_dag()
    network, links = create_dpu_network()
    optimizer = PlacementOptimizer(dag, network, links)

    random_placement = generate_random_placement(dag, optimizer)
    simulator_rand = Simulator(dag, random_placement, network, links, optimizer.router)
    rand_cost = simulator_rand.start_simulation()

    final_placement = optimizer.run_simulated_annealing(
        initial_placement=random_placement,
        initial_temp=1000,
        final_temp=1,
        alpha=0.95,
        steps_per_temp=20,
        heuristic_prob=0.3
    )
    # 我试了一下，好像heuristic还没有纯随机好

    simulator_sa = Simulator(dag, final_placement, network, links, optimizer.router)
    sa_cost = simulator_sa.start_simulation()
    
    if sa_cost == float('inf'):
         print("SA最终方案因超出内存限制而无效！\n")
    else:
        print(f"SA最终成本(Makespan): {sa_cost:.2f} us\n")
