from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

#定义物理的dpu互联网络，默认为胖树结构

@dataclass
class Resource:
    """定义 DPU 内部的资源"""
    id: str # eg. 'dpu1_arm', 'dpu1_dram'
    name: str # 'arm', 'dpa', 'dram', 'ssd', 'nic'
    type: str # 'compute', 'storage' or 'communicate'
    capacity: int = 1
    memory: float = 0.0 # in GB
    # NoC内部传输带宽，假设所有内部连接共享此带宽
    # 注意：这个带宽是用于HEFT算法估算的，DES仿真使用SharedBus的精确带宽
    internal_bandwidth_mbps: float = 800000 # 假设NoC等效带宽 800GB/s
    bandwidth_mbps: float = 0 # in Mbps for storage/nic

@dataclass
class DPU:
    """定义一个 DPU 节点"""
    id: str
    # *** MODIFIED: NoC is now a list of edges (tuples) to represent a graph ***
    noc: List[Tuple[str, str]] = field(default_factory=list) 
    resources: Dict[str, Resource] = field(default_factory=dict)

@dataclass
class Link:
    """定义 DPU 之间的网络连接"""
    id: str
    source_dpu: str
    dest_dpu: str
    bandwidth_gbps: float = 100.0
    latency_us: float = 1.0

#定义数据层面的DAG任务图

@dataclass
class Task:
    """定义 DAG 中的一个节点"""
    id: str
    name: str
    type: str # 'compute' or 'data'

    # if type == 'compute':
    compute_type: str = 'add'

    # if type == 'data':
    du_num: int = 1
    # du_size是每个DU的大小，data_size是总大小. data_size = du_num * du_size
    du_size: float = 0.0
    
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    # data_size是此任务产生的输出数据总量 (MB)
    data_size: float = 0.0 
    rank_u: float = 0.0

#放置方案
Placement = Dict[str, str] # Maps Task ID -> Resource ID