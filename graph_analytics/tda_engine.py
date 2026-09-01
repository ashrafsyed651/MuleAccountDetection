# graph_analytics/tda_engine.py
"""
Offline TDA Engine for detecting circular money laundering cycles.

Extracts suspect subgraphs from Neo4j, computes persistent homology (Betti-1)
using ripser, and pushes the loop confidence scores (features) to Kafka
for ingestion into the Feast Feature Store.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import networkx as nx
import numpy as np
from kafka import KafkaProducer
from neo4j import GraphDatabase, Driver
from ripser import ripser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_FEATURE_TOPIC", "graph-heuristics-features")

def get_driver() -> Driver:
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=5,
    )

def fetch_suspect_subgraph(driver: Driver, limit: int = 1000) -> nx.DiGraph:
    """Extract a subgraph representing recent or suspect transfer activity."""
    # We use inverse transaction volume as distance. To ensure numerical stability,
    # we add a small epsilon to the amount.
    query = """
    MATCH (a:Account)-[r:TRANSFER]->(b:Account)
    WHERE a.account_id IS NOT NULL AND b.account_id IS NOT NULL
    RETURN a.account_id AS source, b.account_id AS target, coalesce(r.amount, 1.0) AS amount
    LIMIT $limit
    """
    graph = nx.DiGraph()
    with driver.session() as session:
        for record in session.run(query, limit=limit):
            # Inverse volume for distance. Larger volume -> shorter distance.
            amount = float(record["amount"])
            dist = 1.0 / (amount + 1e-6)
            graph.add_edge(record["source"], record["target"], weight=dist)
    return graph

def _component_diagrams(graph: nx.Graph) -> list[np.ndarray]:
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return [np.empty((0, 2)), np.empty((0, 2))]
    
    # Use shortest path lengths based on the inverse volume weights.
    distances = nx.floyd_warshall_numpy(graph, nodelist=nodes, weight="weight")
    return ripser(np.asarray(distances), distance_matrix=True, maxdim=1)["dgms"]

def compute_tda_features(graph: nx.DiGraph) -> dict:
    """Compute topological features for each node."""
    undirected = graph.to_undirected()
    features = {
        str(node): {"tda_cycle_count": 0, "tda_h1_persistence": 0.0}
        for node in undirected.nodes()
    }

    for component_nodes in nx.connected_components(undirected):
        component = undirected.subgraph(component_nodes).copy()
        diagrams = _component_diagrams(component)
        
        h1 = diagrams[1]
        lifetimes = h1[:, 1] - h1[:, 0] if len(h1) > 0 else np.array([])
        finite_lifetimes = lifetimes[np.isfinite(lifetimes)]
        persistence = float(finite_lifetimes.sum()) if finite_lifetimes.size > 0 else 0.0
        
        for cycle in nx.cycle_basis(component):
            for account_id in cycle:
                features[str(account_id)]["tda_cycle_count"] += 1
                features[str(account_id)]["tda_h1_persistence"] += persistence

    return features

def publish_tda_features(producer: KafkaProducer, features: dict):
    """Publish TDA features to Kafka."""
    timestamp = datetime.now(timezone.utc).isoformat()
    for account_id, f in features.items():
        if f["tda_cycle_count"] > 0:  # Only publish interesting features to reduce noise
            payload = {
                "account_id": account_id,
                "tda_cycle_count": f["tda_cycle_count"],
                "tda_h1_persistence": f["tda_h1_persistence"],
                "event_timestamp": timestamp,
            }
            producer.send(KAFKA_TOPIC, value=payload)
            logger.info(f"Published TDA payload for account {account_id}: {payload}")
    producer.flush()

def run_tda_pipeline():
    logger.info("Starting TDA pipeline run...")
    driver = get_driver()
    producer = get_producer()
    try:
        graph = fetch_suspect_subgraph(driver)
        logger.info(f"Fetched subgraph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        features = compute_tda_features(graph)
        publish_tda_features(producer, features)
        logger.info("TDA pipeline run complete.")
    except Exception as e:
        logger.error(f"Error in TDA pipeline: {e}")
    finally:
        driver.close()
        producer.close()

if __name__ == "__main__":
    run_tda_pipeline()
