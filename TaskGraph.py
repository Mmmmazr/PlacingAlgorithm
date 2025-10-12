import json
from collections import OrderedDict
from BasicDefinitions import Task

def create_workflow_dag(json_path: str = "TaskGraph.json") -> OrderedDict[str, Task]:
    """创建DAG"""

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = OrderedDict()
    edges = []
    for task in data:
        tasks[task['id']] = Task(id=task['id'], name=task['name'], type=task['type'], data_size=task['data_size'], du_num=task['du_num'], du_size=task.get('du_size', task['data_size'] / task['du_num']))
        for task_id in task['child']:
            edges.append((task['id'], task_id))
    
    for parent_id, child_id in edges:
        if parent_id in tasks and child_id in tasks:
            tasks[parent_id].children.append(child_id)
            tasks[child_id].parents.append(parent_id)

    return tasks

if __name__ == '__main__':
    workflow = create_workflow_dag()
    print(f"Successfully created a DAG with {len(workflow)} nodes.")
    for task_id, task in workflow.items():
        print(f"\n--- Node ID: {task_id} ---")
        print(f"  Name: {task.name}")
        print(f"  Type: {task.type}")
        if task.type == 'data':
            print(f"  DU Num: {task.du_num}")
            print(f"  Data Size: {task.data_size} MB")
        else:
            print(f"  Compute Type: {task.compute_type}")
            print(f"  DU Num: {task.du_num}")
            print(f"  Data Size: {task.data_size} MB")
        print(f"  Parents: {task.parents}")
        print(f"  Children: {task.children}")