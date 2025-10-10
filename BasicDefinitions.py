from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

#定义物理的dpu互联网络
@dataclass
class Resource:
    """DPU内部的资源"""
    id: str # eg. 'dpu1_arm', 'dpu1_dram'
    name: str # 'arm', 'dpa', 'dram', 'ssd', 'nic'
    type: str # 'compute', 'storage' or 'communicate'
    capacity: int = 1
    memory: float = 0.0 # in GB
    internal_bandwidth_mbps: float = 800000 # NoC内部传输带宽， 假设NoC等效带宽 800GB/s 
    bandwidth_mbps: float = 0 # Mbps

@dataclass
class DPU:
    id: str
    noc: List[Tuple[str, str]] = field(default_factory=list) 
    resources: Dict[str, Resource] = field(default_factory=dict)

@dataclass
class Link:
    """DPU间的网络连接"""
    id: str
    source_dpu: str
    dest_dpu: str
    bandwidth_gbps: float = 100.0
    latency_us: float = 1.0

#定义数据层面的DAG任务图
@dataclass
class Task:
    """DAG中的一个节点"""
    id: str
    name: str
    type: str # 'compute' or 'data'

    # if type == 'compute':
    compute_type: str = 'add'

    # 'compute"和'data"都有du_num, du_size, data_size表示出口数据，data_size = du_num * du_size
    du_num: int = 1
    du_size: float = 0.0
    
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    # data_size是此任务产生的输出数据总量 (MB)
    data_size: float = 0.0 
    rank_u: float = 0.0

#放置方案
Placement = Dict[str, str] # Task ID -> Resource ID