"""Persistent-homology helpers for the transaction graph.

This module deliberately has no hard-coded credentials. It can be run once for
an exploratory diagram or imported by the scheduled analytics worker.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from neo4j import GraphDatabase
from persim import plot_diagrams
from ripser import ripser


def fetch_transaction_subgraph(driver, limit: int = 500) -> nx.DiGraph:
    """Fetch a bounded graph so persistent homology remains computationally safe."""
    query = """
    MATCH (a:Account)-[r:TRANSFER]->(b:Account)
    WHERE a.account_id IS NOT NULL AND b.account_id IS NOT NULL
    RETURN a.account_id AS source, b.account_id AS target, coalesce(r.amount, 1.0) AS amount
    LIMIT $limit
    """
    graph = nx.DiGraph()
    with driver.session() as session:
        for record in session.run(query, limit=limit):
            graph.add_edge(record["source"], record["target"], weight=float(record["amount"]))
    return graph


def _component_diagrams(graph: nx.Graph) -> list[np.ndarray]:
    """Return H0/H1 diagrams for one connected component with at least two nodes."""
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return [np.empty((0, 2)), np.empty((0, 2))]

    distances = nx.floyd_warshall_numpy(graph, nodelist=nodes, weight=None)
    return ripser(np.asarray(distances), distance_matrix=True, maxdim=1)["dgms"]


def calculate_topological_features(graph: nx.DiGraph) -> tuple[dict[str, dict[str, float]], list[np.ndarray]]:
    """Produce account-level cycle features and H0/H1 persistence diagrams.

    The H1 lifetime is measured per connected component, then assigned to the
    accounts that participate in a cycle in that component. This avoids treating
    disconnected accounts as artificially close in a single distance matrix.
    """
    undirected = graph.to_undirected()
    features = {
        str(node): {"tda_cycle_count": 0, "tda_h1_persistence": 0.0}
        for node in undirected.nodes()
    }
    h0_diagrams: list[np.ndarray] = []
    h1_diagrams: list[np.ndarray] = []

    for component_nodes in nx.connected_components(undirected):
        component = undirected.subgraph(component_nodes).copy()
        diagrams = _component_diagrams(component)
        h0_diagrams.append(diagrams[0])
        h1_diagrams.append(diagrams[1])

        lifetimes = diagrams[1][:, 1] - diagrams[1][:, 0] if len(diagrams[1]) else np.array([])
        finite_lifetimes = lifetimes[np.isfinite(lifetimes)]
        persistence = float(finite_lifetimes.sum()) if finite_lifetimes.size else 0.0
        for cycle in nx.cycle_basis(component):
            for account_id in cycle:
                features[str(account_id)]["tda_cycle_count"] += 1
                features[str(account_id)]["tda_h1_persistence"] += persistence

    combined = [
        np.vstack(h0_diagrams) if h0_diagrams else np.empty((0, 2)),
        np.vstack(h1_diagrams) if h1_diagrams else np.empty((0, 2)),
    ]
    return features, combined


def save_persistence_diagram(diagrams: list[np.ndarray], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plot_diagrams(diagrams, show=False)
    plt.title("Persistence Diagram of Transaction Subgraph")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline TDA against Neo4j transactions")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--username", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "password"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", default="artifacts/tda_persistence_diagram.png")
    args = parser.parse_args()

    with GraphDatabase.driver(args.uri, auth=(args.username, args.password)) as driver:
        graph = fetch_transaction_subgraph(driver, args.limit)
    features, diagrams = calculate_topological_features(graph)
    save_persistence_diagram(diagrams, args.output)
    cycle_accounts = sum(1 for values in features.values() if values["tda_cycle_count"])
    print(f"Analysed {graph.number_of_nodes()} accounts; {cycle_accounts} participate in graph cycles")
    print(f"Saved persistence diagram to {args.output}")


if __name__ == "__main__":
    main()
