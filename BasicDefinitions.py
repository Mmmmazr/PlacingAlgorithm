from dataclasses import dataclass, field
from typing import List, Dict, Any

#定义物理的dpu互联网络，默认为胖树结构

@dataclass
class Resource:
    """定义 DPU 内部的资源"""
    id: str # eg. '1', '2', ...
    name: str # 'arm', 'dpa', 'dram', 'ssd', 'nic'
    type: str # 'compute', 'storage' or 'communicate'
    # if type == 'compute': cpu or dpa
    capacity: int = 1
    # if type == 'storage': dram or ssd
    memory: float = 0.0 # in GB
    # if type == 'communicate' or 'storage':
    bandwidth_mbps: float = 0 # in Mbps

@dataclass
class DPU:
    """定义一个 DPU 节点"""
    id: str
    resources: Dict[str, Resource] = field(default_factory=dict)
    noc: Dict[str, str] # eg. (resource_id, resource_id)， 表示 DPU 内部资源间的连接

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
    du_size: float = 0.0
    

    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    data_size: float = 0.0 #data_size是output data，计算节点/数据节点都有
    rank_u: float = 0.0

#放置方案
Placement = Dict[str, str] # Maps Task ID -> Resource ID