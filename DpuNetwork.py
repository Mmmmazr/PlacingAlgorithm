from BasicDefinitions import DPU, Resource, Link
from typing import Dict, List, Tuple

def create_dpu_network(num_dpus: int) -> Tuple[Dict[str, DPU], Dict[str, Link]]:
    """
    创建一个DPU网络，每个DPU内部资源全连接，DPU之间也全连接。
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
        resources[f"{dpu_id}_dram"] = Resource(id=f"{dpu_id}_dram", name='dram', type='storage', bandwidth_mbps=240000, memory=128)
        resources[f"{dpu_id}_ssd"] = Resource(id=f"{dpu_id}_ssd", name='ssd', type='storage', bandwidth_mbps=40000, memory=1024)
        resources[f"{dpu_id}_nic"] = Resource(id=f"{dpu_id}_nic", name='nic', type='communicate', bandwidth_mbps=100000)
        
        # *** MODIFIED: 创建DPU内部NoC的全连接图 ***
        noc_edges = []
        resource_ids = list(resources.keys())
        for m in range(len(resource_ids)):
            for n in range(m + 1, len(resource_ids)):
                # 添加双向连接的边
                noc_edges.append((resource_ids[m], resource_ids[n]))
        
        dpus[dpu_id] = DPU(id=dpu_id, resources=resources, noc=noc_edges)

    # 2. 创建 DPU 之间的全连接链路
    dpu_ids = list(dpus.keys())
    for i in range(len(dpu_ids)):
        for j in range(i + 1, len(dpu_ids)):
            dpu_a = dpu_ids[i]
            dpu_b = dpu_ids[j]
            link_id = f"link_{dpu_a}_{dpu_b}"
            links[link_id] = Link(id=link_id, source_dpu=dpu_a, dest_dpu=dpu_b, bandwidth_gbps=200)
            
    print(f"成功创建了一个包含 {len(dpus)} 个DPU和 {len(links)} 条链路的全连接网络。")
    return dpus, links

def print_dpu_network(dpus, links):
    print("\n=== DPU 网络结构 ===")
    for dpu_id, dpu in dpus.items():
        print(f"DPU: {dpu_id}")
        print("  资源:")
        for res_id, res in dpu.resources.items():
            print(f"    {res_id}: {res.name}, 类型: {res.type}")
        print("  NoC连接:")
        for conn, target in dpu.noc:
            print(f"    {conn} -> {target}")
    print("\n链路:")
    for link_id, link in links.items():
        print(f"  {link_id}: {link.source_dpu} <-> {link.dest_dpu}, 带宽: {link.bandwidth_gbps}Gbps")

# 文件末尾添加main函数用于测试
if __name__ == "__main__":
    dpus, links = create_dpu_network(4)
    print_dpu_network(dpus, links)