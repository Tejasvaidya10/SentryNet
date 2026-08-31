import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities


def extract_entity_features(g: nx.Graph) -> pd.DataFrame:
    entity_nodes = [n for n in g.nodes if n[0] == "entity"]

    degree = {n: g.degree(n) for n in entity_nodes}

    component_size = {}
    for component in nx.connected_components(g):
        size = len(component)
        for n in component:
            component_size[n] = size

    entity_proj = nx.bipartite.weighted_projected_graph(g, entity_nodes)

    community_id = {n: -1 for n in entity_nodes}
    if entity_proj.number_of_edges() > 0:
        for i, community in enumerate(louvain_communities(entity_proj, seed=42)):
            for n in community:
                community_id[n] = i

    clustering = (
        nx.clustering(entity_proj)
        if entity_proj.number_of_edges() > 0
        else {n: 0.0 for n in entity_nodes}
    )

    return pd.DataFrame(
        {
            "entity_key": entity_nodes,
            "degree": [degree[n] for n in entity_nodes],
            "component_size": [component_size[n] for n in entity_nodes],
            "community_id": [community_id[n] for n in entity_nodes],
            "clustering_coeff": [clustering.get(n, 0.0) for n in entity_nodes],
        }
    )
