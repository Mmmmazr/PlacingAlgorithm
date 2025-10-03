from collections import OrderedDict
from BasicDefinitions import Task

def create_workflow_dag() -> OrderedDict[str, Task]:
    """
    根据附图的数据处理流程，创建一个符合规范的、节点为数据或操作的DAG。

    - 'data'类型的节点代表图中的数据块。
    - 'compute'类型的节点代表对数据进行处理的操作。
    - 节点的ID以 'd_' 开头表示数据 (data)，以 'c_' 开头表示计算 (compute)。
    - data_size 是对数据大小的估算值，单位为MB。
    """
    tasks = OrderedDict()

    # 1. 定义DAG中所有的节点 (包括数据节点和计算节点)
    
    # --- 输入与Q,K,V生成阶段 ---
    tasks['d_in_ht'] = Task(id='d_in_ht', name='Input Hidden ht', type='data', data_size=16.0, du_num=4)
    tasks['d_kv_cache'] = Task(id='d_kv_cache', name='Input KV Cache', type='data', data_size=128.0, du_num=4)
    
    tasks['c_gen_q'] = Task(id='c_gen_q', name='Generate Q Weights', type='compute', compute_type='linear', data_size=16.0, du_num=4, du_size=4.0)
    tasks['d_q_weights'] = Task(id='d_q_weights', name='Data Q Weights', type='data', data_size=16.0, du_num=4)
    
    tasks['c_gen_wkv_b'] = Task(id='c_gen_wkv_b', name='Generate WKV_B Weights', type='compute', compute_type='linear', data_size=16.0, du_num=4, du_size=4.0)
    tasks['d_wkv_b'] = Task(id='d_wkv_b', name='Data WKV_B Weights', type='data', data_size=16.0, du_num=4)
    
    tasks['c_gen_V'] = Task(id='c_gen_V', name='Generate V from KV Cache', type='compute', compute_type='slice', data_size=8.0, du_num=4, du_size=2.0)
    tasks['d_V'] = Task(id='d_V', name='Data V', type='data', data_size=8.0, du_num=4)

    tasks['c_apply_rope_k'] = Task(id='c_apply_rope_k', name='Apply RoPE to K', type='compute', compute_type='rope', data_size=8.0, du_num=4, du_size=2.0)
    tasks['d_k_pe'] = Task(id='d_k_pe', name='Data k_pe', type='data', data_size=8.0, du_num=4)

    # --- Q路径处理 ---
    tasks['c_apply_rope_q'] = Task(id='c_apply_rope_q', name='Apply RoPE to Q', type='compute', compute_type='rope', data_size=8.0, du_num=4, du_size=2.0)
    tasks['d_q_pe'] = Task(id='d_q_pe', name='Data q_pe', type='data', data_size=8.0, du_num=4)
    
    tasks['c_gen_q_nope'] = Task(id='c_gen_q_nope', name='Generate q_nope', type='compute', compute_type='slice', data_size=8.0, du_num=4, du_size=2.0)
    tasks['d_q_nope'] = Task(id='d_q_nope', name='Data q_nope', type='data', data_size=8.0, du_num=4)

    # --- WKV_B路径处理 (操作 1, 2, 6) ---
    tasks['c_op1_view'] = Task(id='c_op1_view', name='Op 1: View wkv_b', type='compute', compute_type='view', data_size=16.0, du_num=4, du_size=4.0)
    tasks['d_wkv_b_viewed'] = Task(id='d_wkv_b_viewed', name='Data wkv_b Viewed', type='data', data_size=16.0, du_num=4)

    tasks['c_op2_slice_uk'] = Task(id='c_op2_slice_uk', name='Op 2: Slice w_uk', type='compute', compute_type='slice', data_size=8.0, du_num=4, du_size=2.0)
    tasks['d_w_uk'] = Task(id='d_w_uk', name='Data w_uk', type='data', data_size=8.0, du_num=4)

    tasks['c_op6_slice_uv'] = Task(id='c_op6_slice_uv', name='Op 6: Slice w_uv', type='compute', compute_type='slice', data_size=8.0, du_num=4, du_size=2.0)
    tasks['d_w_uv'] = Task(id='d_w_uv', name='Data w_uv', type='data', data_size=8.0, du_num=4)

    # --- Attention分数计算 (操作 3, 4, 5 和加法) ---
    tasks['c_op3_einsum'] = Task(id='c_op3_einsum', name='Op 3: Einsum for score_nope part 1', type='compute', compute_type='einsum', data_size=4.0, du_num=4, du_size=1.0)
    tasks['d_score_nope1'] = Task(id='d_score_nope1', name='Data score_nope part 1', type='data', data_size=4.0, du_num=4)

    tasks['c_op4_einsum'] = Task(id='c_op4_einsum', name='Op 4: Einsum for score_nope part 2', type='compute', compute_type='einsum', data_size=4.0, du_num=4, du_size=1.0)
    tasks['d_score_nope2'] = Task(id='d_score_nope2', name='Data score_nope part 2', type='data', data_size=4.0, du_num=4)

    tasks['c_op5_einsum'] = Task(id='c_op5_einsum', name='Op 5: Einsum for score_pe', type='compute', compute_type='einsum', data_size=4.0, du_num=4, du_size=1.0)
    tasks['d_score_pe'] = Task(id='d_score_pe', name='Data score_pe', type='data', data_size=4.0, du_num=4)

    tasks['c_add_scores'] = Task(id='c_add_scores', name='Add Scores', type='compute', compute_type='add', data_size=4.0, du_num=4, du_size=1.0)
    tasks['d_scores_sum'] = Task(id='d_scores_sum', name='Data Scores Summed', type='data', data_size=4.0, du_num=4)
    
    # --- Softmax和输出计算 (操作 7 和 wo) ---
    tasks['c_softmax'] = Task(id='c_softmax', name='Softmax', type='compute', compute_type='softmax', data_size=4.0, du_num=4, du_size=1.0)
    tasks['d_scores_softmax'] = Task(id='d_scores_softmax', name='Data Scores after Softmax', type='data', data_size=4.0, du_num=4)

    tasks['c_op7_einsum'] = Task(id='c_op7_einsum', name='Op 7: Einsum for Output', type='compute', compute_type='einsum', data_size=16.0, du_num=4, du_size=4.0)
    tasks['d_pre_output'] = Task(id='d_pre_output', name='Data Pre-Output', type='data', data_size=16.0, du_num=4)

    tasks['c_op_wo'] = Task(id='c_op_wo', name='Op wo: Linear Output', type='compute', compute_type='linear', data_size=16.0, du_num=4, du_size=4.0)
    tasks['d_out_ht'] = Task(id='d_out_ht', name='Output Hidden ht', type='data', data_size=16.0, du_num=4)

    # 2. 定义节点之间的依赖关系 (父->子)
    edges = [
        ('d_in_ht', 'c_gen_q'),
        ('d_in_ht', 'c_gen_wkv_b'),
        ('c_gen_q', 'd_q_weights'),
        ('c_gen_wkv_b', 'd_wkv_b'),
        
        ('d_kv_cache', 'c_gen_V'),
        ('d_kv_cache', 'c_apply_rope_k'),
        ('c_gen_V', 'd_V'),
        ('c_apply_rope_k', 'd_k_pe'),

        ('d_q_weights', 'c_apply_rope_q'),
        ('d_q_weights', 'c_gen_q_nope'),
        ('c_apply_rope_q', 'd_q_pe'),
        ('c_gen_q_nope', 'd_q_nope'),

        ('d_wkv_b', 'c_op1_view'),
        ('c_op1_view', 'd_wkv_b_viewed'),
        ('d_wkv_b_viewed', 'c_op2_slice_uk'),
        ('d_wkv_b_viewed', 'c_op6_slice_uv'),
        ('c_op2_slice_uk', 'd_w_uk'),
        ('c_op6_slice_uv', 'd_w_uv'),

        ('d_q_nope', 'c_op3_einsum'),
        ('d_w_uk', 'c_op3_einsum'),
        ('c_op3_einsum', 'd_score_nope1'),

        ('d_q_nope', 'c_op4_einsum'),
        ('d_kv_cache', 'c_op4_einsum'),
        ('c_op4_einsum', 'd_score_nope2'),

        ('d_q_pe', 'c_op5_einsum'),
        ('d_k_pe', 'c_op5_einsum'),
        ('c_op5_einsum', 'd_score_pe'),

        ('d_score_nope1', 'c_add_scores'),
        ('d_score_nope2', 'c_add_scores'),
        ('d_score_pe', 'c_add_scores'),
        ('c_add_scores', 'd_scores_sum'),

        ('d_scores_sum', 'c_softmax'),
        ('c_softmax', 'd_scores_softmax'),
        
        # 简化：操作7使用Softmax的输出和V、w_uv。这里假设它们被一个计算节点消耗
        ('d_scores_softmax', 'c_op7_einsum'),
        ('d_V', 'c_op7_einsum'),
        ('d_w_uv', 'c_op7_einsum'),
        ('c_op7_einsum', 'd_pre_output'),

        ('d_pre_output', 'c_op_wo'),
        ('c_op_wo', 'd_out_ht'),
    ]

    # 3. 根据依赖关系构建图
    for parent_id, child_id in edges:
        if parent_id in tasks and child_id in tasks:
            tasks[parent_id].children.append(child_id)
            tasks[child_id].parents.append(parent_id)

    return tasks

if __name__ == '__main__':
    # 用于测试的简单打印功能
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