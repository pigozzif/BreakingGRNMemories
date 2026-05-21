import networkx as nx
import numpy as np
import sbmltoodejax

from utils import set_seed

NETWORK_PROPERTIES = ["nodes.number",
                      "edges.number",
                      "degree.in.mean",
                      "degree.in.std",
                      "degree.out.mean",
                      "degree.out.std",
                      "betweenness.mean",
                      "betweenness.std",
                      "pagerank.mean",
                      "pagerank.std",
                      "h.mean",
                      "h.std",
                      "a.mean",
                      "a.std"]


def sbml2networkx(biomodel_data):
    graph = nx.DiGraph()
    species = [k for k, v in biomodel_data.species.items() if not v.isConstant]
    graph.add_nodes_from(species)
    for reaction in biomodel_data.reactions.values():
        for s in filter(lambda x: x[0] == -1, reaction.reactants):
            for t in filter(lambda x: x[0] == 1, reaction.reactants):
                graph.add_edge(s[1], t[1])
    return graph


def get_graph_features(graph):
    in_degrees = [graph.in_degree(node) for node in graph.nodes]
    out_degrees = [graph.out_degree(node) for node in graph.nodes]
    betweenness = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank(graph)
    h, a = nx.hits(graph)
    return [graph.number_of_nodes(),
            graph.number_of_edges(),
            np.mean(in_degrees),
            np.std(in_degrees),
            np.mean(out_degrees),
            np.std(out_degrees),
            np.mean(list(betweenness.values())),
            np.std(list(betweenness.values())),
            np.mean(list(pagerank.values())),
            np.std(list(pagerank.values())),
            np.mean(list(h.values())),
            np.std(list(h.values())),
            np.mean(list(a.values())),
            np.std(list(a.values()))]


def get_network_properties(biomodel_id):
    model_data = sbmltoodejax.parse.ParseSBMLFile(f"data/biomodel_{biomodel_id}.xml")
    network = sbml2networkx(biomodel_data=model_data)
    features = get_graph_features(graph=network)
    return features


if __name__ == "__main__":
    set_seed(0)
    with open("networks.txt", "w") as file:
        file.write(";".join(NETWORK_PROPERTIES) + "\n")
        for model_id in [2, 3, 4, 5, 6, 10, 69, 16, 17, 22, 21, 23, 26, 27, 483, 29, 31, 631, 203, 204, 209, 210, 275,
                         39, 50, 35, 36, 37, 38]:
            feats = get_network_properties(biomodel_id=model_id)
            file.write(";".join([str(f) for f in feats]) + "\n")
