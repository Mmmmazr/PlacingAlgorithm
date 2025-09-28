from BasicDefinitions import DPU, Resource, Link
from typing import Dict, List, Tuple

def create_dpu_network(num_dpus: int) -> Tuple[Dict[str, DPU], Dict[str, Link]]:
    """
    一个修复版本的 DPU 网络创建函数。
    它会正确地创建指定数量的 DPU 及其资源，并建立一个全连接的网络。
    """
    dpus = {}
    links = {}

    # 1. 创建 num_dpus 个 DPU 实体
    for i in range(1, num_dpus + 1):
        dpu_id = f"dpu{i}"
        resources = {}
        # 为每个 DPU 添加计算、存储和网络资源
        resources[f"{dpu_id}_arm"] = Resource(id=f"{dpu_id}_arm", name='arm', type='compute', capacity=16)
        resources[f"{dpu_id}_dpa"] = Resource(id=f"{dpu_id}_dpa", name='dpa', type='compute', capacity=256)
        # 不同的DPU可以有不同的存储带宽或容量，这里为了简单设为一致
        resources[f"{dpu_id}_dram"] = Resource(id=f"{dpu_id}_dram", name='dram', type='storage', bandwidth_mbps=240000, memory=128)
        resources[f"{dpu_id}_ssd"] = Resource(id=f"{dpu_id}_ssd", name='ssd', type='storage', bandwidth_mbps=40000, memory=1024)
        # 关键：确保 NIC 的 ID 格式与 Simulator 中使用的格式一致
        resources[f"{dpu_id}_nic"] = Resource(id=f"{dpu_id}_nic", name='nic', type='communicate', bandwidth_mbps=100000)
        
        # DPU内部连接 (NoC) - 这里简化，实际可以更复杂
        noc = {}
        
        dpus[dpu_id] = DPU(id=dpu_id, resources=resources, noc=noc)

    # 2. 创建 DPU 之间的全连接链路
    dpu_ids = list(dpus.keys())
    for i in range(len(dpu_ids)):
        for j in range(i + 1, len(dpu_ids)):
            dpu_a = dpu_ids[i]
            dpu_b = dpu_ids[j]
            link_id = f"link_{dpu_a}_{dpu_b}"
            links[link_id] = Link(id=link_id, source_dpu=dpu_a, dest_dpu=dpu_b, bandwidth_gbps=200)
            
    # print(f"成功创建了一个包含 {len(dpus)} 个DPU和 {len(links)} 条链路的全连接网络。")
    return dpus, links