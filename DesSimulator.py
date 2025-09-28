import simpy
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict
from BasicDefinitions import Task, DPU, Link, Placement
from PlacementAlgorithm import Router

class SharedBus:
    def __init__(self, env, name, bandwidth_mbps):
        self.env = env
        self.name = name
        # Mbps to MB/us (1e6 bytes/s = 1 MB/s; 1e6 us/s = 1 s; 1 MB/s = 1 MB/1e6 us = 1e-6 MB/us)
        # Mbps / 8 = MB/s
        self.bandwidth_mbps = bandwidth_mbps
        self.bandwidth_mb_us = (bandwidth_mbps / 8) / 1e6 # MB/s to MB/us (MB/s * 10^-6)
        self.active_transfers = []
        self.process = env.process(self.run())
        self.wakeup_event = env.event()
        
    def run(self):
        time_of_last_update = self.env.now
        while True:
            if not self.active_transfers:
                # 等待新的传输请求
                yield self.wakeup_event
                self.wakeup_event = self.env.event()
                time_of_last_update = self.env.now
                continue
            
            try:
                effective_bw = self.bandwidth_mb_us / len(self.active_transfers)
                if effective_bw <= 1e-12: # 防止除以零或带宽极低
                    yield self.env.timeout(1e9) # 长时间等待，直到带宽可用
                    continue
                    
                times_to_finish = [t['remaining'] / effective_bw for t in self.active_transfers]
                time_to_next_completion = min(times_to_finish)
                
                # 等待直到下一个事件完成
                yield self.env.timeout(time_to_next_completion)
                
                # 更新状态
                time_passed = self.env.now - time_of_last_update
                data_transferred = effective_bw * time_passed
                
                new_active_list = []
                for t in self.active_transfers:
                    t['remaining'] -= data_transferred
                    if t['remaining'] > 1e-9:
                        new_active_list.append(t)
                    else:
                        t['event'].succeed()
                        
                self.active_transfers = new_active_list
                time_of_last_update = self.env.now
                
            except simpy.Interrupt:
                # 处理中断（新的传输到达）
                time_passed = self.env.now - time_of_last_update
                num_transfers_before_interrupt = len(self.active_transfers) - 1
                
                if num_transfers_before_interrupt > 0:
                    old_effective_bw = self.bandwidth_mb_us / num_transfers_before_interrupt
                    data_transferred = old_effective_bw * time_passed
                    
                    # 只更新中断前已经在列表中的传输
                    for t in self.active_transfers[:-1]: 
                        t['remaining'] -= data_transferred
                        
                time_of_last_update = self.env.now
                
    def transfer(self, data_size_mb):
        if data_size_mb <= 1e-9: return self.env.timeout(0)
        
        completion_event = self.env.event()
        was_idle = (len(self.active_transfers) == 0)
        
        self.active_transfers.append({'remaining': data_size_mb, 'event': completion_event})
        
        if was_idle:
            if not self.wakeup_event.triggered:
                self.wakeup_event.succeed()
        else:
            # 中断运行中的进程，使其重新计算分配的带宽和下一个超时时间
            self.process.interrupt() 
            
        return completion_event

# --- Simulator 类 (主要修复区域) ---

class Simulator:
    def __init__(self, dag: Dict[str, Task], placement: Placement,
                 network: Dict[str, DPU], links: Dict[str, Link], router: Router):
        self.env = simpy.Environment()
        self.dag = dag
        self.placement = placement
        self.network = network
        self.links = links
        self.router = router
        self.resources = {}

        # 1. 初始化所有物理资源 (Compute, Storage, NICs)
        for dpu in network.values():
            for res in dpu.resources.values():
                if res.type == 'compute':
                    # 为展开的计算核心创建独占资源
                    for i in range(res.capacity):
                        self.resources[f"{res.id}_{i}"] = simpy.Resource(self.env, capacity=1)
                elif res.bandwidth_mbps > 0:
                    # 为存储总线/NIC/其他总线创建共享总线资源
                    self.resources[res.id] = SharedBus(self.env, res.id, res.bandwidth_mbps)
        
        # 2. 初始化外部网络链路
        for link in links.values():
            bandwidth_mbps = link.bandwidth_gbps * 1000 / 8 # GB/s * 1000 = MB/s
            link_id_forward = f"link_{link.source_dpu}_{link.dest_dpu}"
            link_id_backward = f"link_{link.dest_dpu}_{link.source_dpu}"
            
            # 外部链路是双向的，但我们用一个SharedBus来模拟双向带宽，通常不准确，
            # 但这里为了简化，假设一个SharedBus代表链路的总带宽资源
            self.resources[link_id_forward] = SharedBus(self.env, link_id_forward, bandwidth_mbps)
            # 确保反向路径也能找到这个资源
            self.resources[link_id_backward] = self.resources[link_id_forward] 

        # 3. 关键：为每个任务的每个数据单元(DU)创建完成事件
        self.task_du_done_events: Dict[str, List[simpy.Event]] = {}
        for task_id, task in dag.items():
            num_dus = task.du_num if task.du_num > 0 else 1
            self.task_du_done_events[task_id] = [self.env.event() for _ in range(num_dus)]
            
        # 4. 关键：DU到达目标计算节点的 Store (解决动态事件注册风险)
        # 结构: self.du_arrival_stores[SourceTaskID][DestTaskID][DU_Index] -> simpy.Store
        self.du_arrival_stores = defaultdict(lambda: defaultdict(dict))

    def _get_resource_dpu_id(self, resource_id: str) -> str:
        """根据资源ID查找它所属的DPU ID"""
        for dpu_id, dpu in self.network.items():
            # 检查资源是否在 dpu.resources 中
            if resource_id in dpu.resources:
                return dpu_id
            # 检查资源是否是展开的 compute core ID 的一部分 (如 'dpu1_core_0_0')
            if resource_id.startswith(f"{dpu_id}_"):
                 return dpu_id
        raise ValueError(f"Resource ID {resource_id} not found in any DPU.")

    def run(self):
        # 为DAG中的每个任务启动一个仿真进程
        for task_id in self.dag:
            self.env.process(self._task_process_router(task_id))
        
        # 找到所有最终的输出数据节点 (即没有子节点的任务)
        final_tasks = [tid for tid, task in self.dag.items() if not task.children]
        if not final_tasks: return 0.0

        # 等待所有最终任务的所有DU都完成
        final_events = []
        for tid in final_tasks:
            final_events.extend(self.task_du_done_events[tid])
            
        # run() 应该返回一个 SimPy Event 供 env.run() 使用
        return simpy.AllOf(self.env, final_events)

    def start_simulation(self):
        # env.run() 接受一个进程/事件，并等待其完成
        self.env.run(self.run()) 
        return self.env.now

    def _task_process_router(self, task_id: str):
        """根据任务类型，分派给相应的处理进程"""
        task = self.dag[task_id]
        if task.type == 'compute':
            yield self.env.process(self._compute_task_process(task_id))
        elif task.type == 'data':
            yield self.env.process(self._data_task_process(task_id))

    def _compute_task_process(self, task_id: str):
        """仿真一个计算任务的完整生命周期（接收DU->计算DU->完成）"""
        task = self.dag[task_id]
        num_dus = task.du_num if task.du_num > 0 else 1
        
        # 1. 启动所有数据传输工作进程
        # 每个父节点都有一个独立的worker从其位置传输数据到本计算节点
        for parent_id in task.parents:
            # 传输 worker 将数据传输到 Store，而不是直接操作事件
            self.env.process(
                self._data_transfer_worker(source_id=parent_id, dest_id=task_id)
            )

        # 2. 流水线式处理每个数据单元 (DU)
        compute_workers = []
        for i in range(num_dus):
            # 为每个DU启动一个单独的计算worker
            worker = self.env.process(self._compute_worker(task_id, i))
            compute_workers.append(worker)
        
        # 等待本计算任务所有DU的完成事件
        yield simpy.AllOf(self.env, self.task_du_done_events[task_id])

    def _compute_worker(self, task_id: str, du_index: int):
        """一个worker，负责处理单个DU的计算"""
        task = self.dag[task_id]
        
        # A. 等待所有父节点的第 du_index 个DU到达 (通过 Store 安全等待)
        arrival_gets = []
        for parent_id in task.parents:
            # 如果 Store 不存在，创建它。Store 允许发送方在接收方请求前存放数据。
            if du_index not in self.du_arrival_stores[parent_id][task_id]:
                self.du_arrival_stores[parent_id][task_id][du_index] = simpy.Store(self.env, capacity=1)
                
            arrival_store = self.du_arrival_stores[parent_id][task_id][du_index]
            arrival_gets.append(arrival_store.get())
        
        # 等待所有父节点的 DU 到达 Token
        if arrival_gets:
            yield simpy.AllOf(self.env, arrival_gets)

        # B. DU已到达，开始计算
        compute_res_id = self.placement[task_id]
        # 假设 compute_res_id 包含了 DPU ID 和 core index, e.g., 'dpu1_core_0_0'
        compute_resource = self.resources[compute_res_id] 
        
        # 计算工作量模拟
        workload = {'linear': 10, 'slice': 2, 'rope': 15, 'view': 1, 'einsum': 25, 'add': 2, 'softmax': 8}
        du_compute_time = workload.get(task.compute_type, 5) / (task.du_num or 1)
        
        with compute_resource.request() as req:
            yield req
            yield self.env.timeout(du_compute_time)
        
        # C. 计算完成，触发本任务本DU的完成事件
        self.task_du_done_events[task_id][du_index].succeed()

    def _data_task_process(self, task_id: str):
        """仿真一个数据任务的完整生命周期（等待输入DU->写入存储->完成）"""
        task = self.dag[task_id]
        num_dus = task.du_num if task.du_num > 0 else 1
        
        # --- 修复点 3: 初始数据节点 DU 顺序发出 ---
        if not task.parents:
            du_generation_interval = 0.5 # 模拟每个DU生成之间的间隔
            for i in range(num_dus):
                yield self.env.timeout(du_generation_interval) 
                self.task_du_done_events[task_id][i].succeed()
            return
        # ----------------------------------------
        
        # 非初始数据任务： 流水线式处理每个DU的写入
        for i in range(num_dus):
            self.env.process(self._data_worker(task_id, i))
            
        yield simpy.AllOf(self.env, self.task_du_done_events[task_id])

    def _data_worker(self, task_id: str, du_index: int):
        """一个worker，负责将单个DU写入存储"""
        task = self.dag[task_id]
        
        # A. 等待所有父节点的第 du_index 个DU完成 (计算或数据写入完成)
        # 修复点 2: 父节点可以是任意类型
        parent_du_done_events = [self.task_du_done_events[pid][du_index] for pid in task.parents]
        yield simpy.AllOf(self.env, parent_du_done_events)
        
        # B. 将聚合后的DU写入目标存储
        storage_res_id = self.placement[task_id]
        storage_bus = self.resources[storage_res_id]
        du_size = task.data_size / (task.du_num or 1)
        
        if du_size > 0:
            yield storage_bus.transfer(du_size)
            
        # C. 写入完成，触发本任务本DU的完成事件
        self.task_du_done_events[task_id][du_index].succeed()

    def _data_transfer_worker(self, source_id: str, dest_id: str):
        """一个worker，负责将一个数据源的所有DU，通过网络，流水线式地传输到目标节点"""
        
        source_task = self.dag[source_id]
        dest_task = self.dag[dest_id]
        
        num_dus = source_task.du_num if source_task.du_num > 0 else 1
        du_size = source_task.data_size / num_dus if num_dus > 0 else 0
        
        if du_size <= 1e-9: # 无效数据量，仍然需要触发下游事件
            for i in range(num_dus):
                if i not in self.du_arrival_stores[source_id][dest_id]:
                    self.du_arrival_stores[source_id][dest_id][i] = simpy.Store(self.env, capacity=1)
                yield self.du_arrival_stores[source_id][dest_id][i].put(True)
            return

        # 确定源和目标DPU ID (支持计算节点接收计算节点信息)
        source_res_id = self.placement[source_id]
        dest_res_id = self.placement[dest_id]
        
        # 修复点 2/4: 通过放置查找 DPU ID
        source_dpu_id = self._get_resource_dpu_id(source_res_id)
        dest_dpu_id = self._get_resource_dpu_id(dest_res_id)
        
        # 确定通信路径上的资源
        comm_path_resources = []
        
        # 1. 源DRAM/存储总线 (读取操作)
        # 假设数据任务 placement 对应 DRAM/SSD 总线
        # 假设计算任务 placement 对应某个核心，但源数据本身总有一个存储位置
        source_storage_id = source_res_id if source_task.type == 'data' else next(
            r.id for r in self.network[source_dpu_id].resources.values() if r.type == 'storage' and r.name == 'dram'
        ) 
        comm_path_resources.append(self.resources[source_storage_id]) 

        # 2. DPU 间传输 (如果不在同一个 DPU)
        if source_dpu_id != dest_dpu_id:
            # 2a. 源 NIC
            source_nic_id = f"{source_dpu_id}_nic" # 假设 NIC ID 命名规则
            comm_path_resources.append(self.resources[source_nic_id])
            
            # 2b. 外部网络链路 (由 Router 确定路径)
            dpu_path = self.router.get_path(source_dpu_id, dest_dpu_id)
            if len(dpu_path) > 1: # 确保路径中包含链路
                for i in range(len(dpu_path) - 1):
                    # 修复点 3: 路由路径资源查找
                    dpu_src, dpu_dst = dpu_path[i], dpu_path[i+1]
                    link_id_fwd = f"link_{dpu_src}_{dpu_dst}"
                    link_id_bwd = f"link_{dpu_dst}_{dpu_src}"
                    
                    if link_id_fwd in self.resources:
                        comm_path_resources.append(self.resources[link_id_fwd])
                    elif link_id_bwd in self.resources:
                        comm_path_resources.append(self.resources[link_id_bwd])
                    else:
                        print(f"Warning: Link not found for path segment {dpu_src}->{dpu_dst}")

            # 2c. 目标 NIC
            dest_nic_id = f"{dest_dpu_id}_nic" # 假设 NIC ID 命名规则
            comm_path_resources.append(self.resources[dest_nic_id])
        
        # 3. 目标 DPU 内部总线/存储总线 (写入目标)
        # 简化：数据最终写入目标 DPU 的 DRAM
        dest_storage_id = next(r.id for r in self.network[dest_dpu_id].resources.values() if r.type == 'storage' and r.name == 'dram')
        comm_path_resources.append(self.resources[dest_storage_id])

        # 流水线式传输每个DU
        for i in range(num_dus):
            # A. 等待源数据DU就绪 (计算/数据写入完成)
            yield self.task_du_done_events[source_id][i]
            
            # B. 沿路径传输DU，会产生争用
            for res_bus in comm_path_resources:
                yield res_bus.transfer(du_size)
            
            # C. DU到达目标，触发对应的到达 Store (发送 Token)
            # 修复点 2: 使用 Store 安全通信
            if i not in self.du_arrival_stores[source_id][dest_id]:
                self.du_arrival_stores[source_id][dest_id][i] = simpy.Store(self.env, capacity=1)
                
            arrival_store = self.du_arrival_stores[source_id][dest_id][i]
            # put() 是 SimPy 的 Event，需要 yield 来等待操作完成 (即数据放入 Store)
            yield arrival_store.put(True) 
