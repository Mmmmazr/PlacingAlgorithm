import json
from BasicDefinitions import DPU, Resource, Link
from typing import Dict, List, Tuple

def create_dpu_network(json_path: str = r'C:\code\PlacingAlgorithm\DpuNetwork.json') -> Tuple[Dict[str, DPU], Dict[str, Link]]:
    """
    创建DPU网络，每个DPU内部资源全连接，DPU之间也全连接。
    """
    dpus = {}
    links = {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for dpu in data:
        resources = {}
        for resource in dpu['resources']:
            type = resource['type']
            if type == 'compute':
                resources[resource['resource_id']] = Resource(id=resource['resource_id'], name=resource['name'], type=type, capacity=resource['capacity'])
            elif type == 'storage':
                resources[resource['resource_id']] = Resource(id=resource['resource_id'], name=resource['name'], type=type, memory=resource['memory'], bandwidth_MBps=resource['bandwidth_MBps'])
            else:
                resources[resource['resource_id']] = Resource(id=resource['resource_id'], name=resource['name'], type=type, bandwidth_MBps=resource['bandwidth_MBps'])

        noc_edges = []
        for edge in dpu['noc']:
            noc_edges.append((edge[0], edge[1]))

        dpus[dpu['dpu_id']] = DPU(id=dpu['dpu_id'], resources=resources, noc=noc_edges)

        for neighbor in dpu['neighbors']:
            link_id = f"link_{dpu['dpu_id']}_{neighbor['id']}"
            links[link_id] = Link(id=link_id, source_dpu=dpu['dpu_id'], dest_dpu=neighbor['id'], bandwidth_MBps=neighbor['bandwidth_MBps'])
            
    # print(f"成功创建了一个包含 {len(dpus)} 个DPU和 {len(links)} 条链路的全连接网络。")
    return dpus, links

# test
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

if __name__ == "__main__":
    dpus, links = create_dpu_network()
    print_dpu_network(dpus, links)