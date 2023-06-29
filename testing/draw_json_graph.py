import json
import networkx as nx
import matplotlib.pyplot as plt
from itertools import count


og_graph_path = "/home/mihir/source_code/Subdue/testing/cmo_graph.json"
pattern1_path = "/home/mihir/source_code/Subdue/testing/three_bwt-pattern-1.json"
compressed_graph_1_path = "/home/mihir/source_code/Subdue/testing/three_bwt-compressed-1.json"
pattern2_path = "/home/mihir/source_code/Subdue/testing/three_bwt-pattern-2.json"
compressed_graph_2_path = "/home/mihir/source_code/Subdue/testing/three_bwt-compressed-2.json"

def subdue_json_to_undirected_nx_graph(subdue_json_path):
    """WARNING: ignores directed edges and timestamp on purpose"""

    with open(subdue_json_path, 'r') as subdue_json_file:
        subdue_format = json.load(subdue_json_file)

    graph = nx.Graph()
    for vertex_or_edge in subdue_format:
        if list(vertex_or_edge.keys()) == ['vertex']:
            node_attributes_loop = vertex_or_edge['vertex']['attributes']
            # graph.add_node(
            #     vertex_or_edge['vertex']['attributes']['label'],
            #     **node_attributes_loop,
            # )
            graph.add_node(
                vertex_or_edge['vertex']['id'],
                **node_attributes_loop,
            )
        elif list(vertex_or_edge.keys()) == ['edge']:
            edge_attributes_loop = vertex_or_edge['edge']['attributes']
            graph.add_edge(
                u_of_edge=vertex_or_edge['edge']['source'],
                v_of_edge=vertex_or_edge['edge']['target'],
                **edge_attributes_loop,
            )
        else:
            raise ValueError('Invalid entry type')

    return graph

def drawGrpah(graph, color_attr=None):
    if color_attr == None:
        nx.draw(graph)
        plt.show()
    else:
        groups = set(nx.get_node_attributes(graph,color_attr).values())
        node_id_list = list(graph.nodes)
        node_id_dict = {}
        for nid in node_id_list:
            node_id_dict[nid] = nid
        mapping = dict(zip(sorted(groups),count()))
        nodes = graph.nodes()
        colors = [mapping[graph.nodes[n][color_attr]] for n in nodes]

        # drawing nodes and edges separately so we can capture collection for colobar
        pos = nx.spring_layout(graph)
        ec = nx.draw_networkx_edges(graph, pos, alpha=0.2)
        nc = nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)
        lbl = nx.draw_networkx_labels(graph, pos, labels=node_id_dict, font_size=18)
        plt.axis('off')
        plt.show()


og_graph = subdue_json_to_undirected_nx_graph(og_graph_path)
drawGrpah(og_graph, 'label')

# pattern1 = subdue_json_to_undirected_nx_graph(pattern1_path)
# drawGrpah(pattern1, 'label')

# compressed_graph_1 = subdue_json_to_undirected_nx_graph(compressed_graph_1_path)
# drawGrpah(compressed_graph_1, 'label')

# pattern2 = subdue_json_to_undirected_nx_graph(pattern2_path)
# drawGrpah(pattern2, 'label')

# compressed_graph_2 = subdue_json_to_undirected_nx_graph(compressed_graph_2_path)
# drawGrpah(compressed_graph_2, 'label')

# nx.draw(og_graph, with_labels=True)
# plt.show()
# pattern1 = subdue_json_to_undirected_nx_graph(pattern1_path)
# nx.draw(pattern1)
# plt.show()
# compressed_graph = subdue_json_to_undirected_nx_graph(compressed_graph_path)
# nx.draw(compressed_graph)
# plt.show()