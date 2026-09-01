# graph_analytics/heuristics.py
"""
Heuristics module for real‑time graph feature extraction.

- Computes Strongly Connected Components (Tarjan's algorithm) via Neo4j GDS.
- Computes Louvain community detection via Neo4j GDS.
- Retrieves node degree.
- Publishes metrics to a Kafka topic so that the Feast CDC consumer can ingest them.

Assumes the Neo4j instance is reachable via the standard bolt URL and that the
Kafka broker is accessible at ``kafka:29092`` (same as the existing consumer).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from kafka import KafkaProducer
from neo4j import GraphDatabase, Driver, Transaction

# Configure logger – the surrounding project already uses ``print`` in the
# consumer, but a proper logger gives us more flexibility.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Neo4j connection handling
# ---------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")

def _get_driver() -> Driver:
    """Create a Neo4j driver instance.

    The driver is lightweight; we keep a single global instance for the
    lifetime of the process.  In production you would want a more robust pool
    and proper shutdown handling.
    """
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ---------------------------------------------------------------------------
# Kafka producer handling
# ---------------------------------------------------------------------------

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_FEATURE_TOPIC", "graph-heuristics-features")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    retries=5,
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _window_timestamp_range(hours: int = 24) -> Tuple[int, int]:
    """Return the Unix‑epoch millisecond range representing the past *hours*.

    Neo4j timestamps are stored as ``datetime`` values; the GDS procedures we
    call accept a ``timestamp`` property that we filter on.  By returning the
    lower and upper bounds we keep the Cypher concise.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    # Convert to epoch milliseconds – Neo4j ``datetime`` can be compared to a
    # number representing epoch millis.
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    start_ms = int((start - epoch).total_seconds() * 1000)
    end_ms = int((now - epoch).total_seconds() * 1000)
    return start_ms, end_ms

# ---------------------------------------------------------------------------
# Core analytics functions
# ---------------------------------------------------------------------------

def compute_scc_and_louvain(tx: Transaction, start_ts: int, end_ts: int) -> List[Dict]:
    """Run SCC (Tarjan) and Louvain community detection on the 24‑hour window.

    The GDS library expects a *projected* graph.  For simplicity we create a
    transient in‑memory graph for each run.  The projection filters edges based
    on a ``timestamp`` property that must fall within ``[start_ts, end_ts]``.
    """
    # 1️⃣ Project a temporary graph called ``tempGraph``.
    projection_cypher = """
    CALL gds.graph.project.cypher(
        'tempGraph',
        'MATCH (n:Account) RETURN id(n) AS id',
        '''
        MATCH (a:Account)-[r:TRANSFER]->(b:Account)
        WHERE r.timestamp >= $start_ts AND r.timestamp <= $end_ts
        RETURN id(a) AS source, id(b) AS target, r.amount AS weight
        '''
    )
    YIELD graphName, nodeCount, relationshipCount
    """
    tx.run(projection_cypher, start_ts=start_ts, end_ts=end_ts).consume()

    # 2️⃣ Run SCC (Tarjan) – the GDS procedure returns a componentId per node.
    scc_cypher = """
    CALL gds.beta.scc
        .stats('tempGraph')
        YIELD componentCount, nodeCount, relationshipCount, maxComponentSize, minComponentSize, averageComponentSize
    """
    scc_stats = tx.run(scc_cypher).single()

    # 3️⃣ Run Louvain – we request the communityId per node.
    louvain_cypher = """
    CALL gds.louvain
        .write('tempGraph', {
            writeProperty: 'louvainCommunityId'
        })
        YIELD communityCount, modularity, ranIterations
    """
    tx.run(louvain_cypher).consume()

    # 4️⃣ Retrieve per‑node metrics (degree, SCC id, Louvain id).
    result_cypher = """
    MATCH (n:Account)
    OPTIONAL MATCH (n)-[r:TRANSFER]->()
    RETURN
        n.account_id AS account_id,
        size((n)-[:TRANSFER]-()) AS node_degree,
        gds.util.sccComponentId('tempGraph', id(n)) AS scc_cluster_id,
        n.louvainCommunityId AS louvain_community_id
    """
    records = tx.run(result_cypher)
    metrics = []
    for rec in records:
        metrics.append({
            "account_id": rec["account_id"],
            "node_degree": rec["node_degree"],
            "scc_cluster_id": rec["scc_cluster_id"],
            "louvain_community_id": rec["louvain_community_id"],
        })

    # 5️⃣ Drop the temporary graph to keep Neo4j tidy.
    tx.run("CALL gds.graph.drop('tempGraph') YIELD graphName").consume()
    return metrics

def publish_metrics(metrics: List[Dict]) -> None:
    """Publish each node's metrics to Kafka.

    The payload format mirrors what the Feast CDC consumer expects – a flat dict
    with the feature values plus a server‑side timestamp.
    """
    for metric in metrics:
        payload = {
            "account_id": metric["account_id"],
            "node_degree": metric["node_degree"],
            "scc_community_id": metric["scc_cluster_id"],
            "louvain_community_id": metric["louvain_community_id"],
            "event_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        producer.send(KAFKA_TOPIC, value=payload)
        logger.info(f"Published heuristics payload for account {metric['account_id']}")
    producer.flush()

def run_heuristics_loop(poll_interval_seconds: int = 300) -> None:
    """Continuously compute and publish heuristics.

    The function runs forever; in a containerised deployment you would run it as
    the main process.  It sleeps ``poll_interval_seconds`` between iterations.
    """
    driver = _get_driver()
    try:
        while True:
            start_ts, end_ts = _window_timestamp_range(hours=24)
            with driver.session() as session:
                metrics = session.read_transaction(compute_scc_and_louvain, start_ts, end_ts)
                if metrics:
                    publish_metrics(metrics)
                else:
                    logger.warning("No metrics returned for the current window.")
            logger.info(f"Heuristics run completed – sleeping {poll_interval_seconds}s")
            time.sleep(poll_interval_seconds)
    finally:
        driver.close()

if __name__ == "__main__":
    # When executed directly we run a single iteration for quick debugging.
    driver = _get_driver()
    start_ts, end_ts = _window_timestamp_range(hours=24)
    with driver.session() as session:
        metrics = session.read_transaction(compute_scc_and_louvain, start_ts, end_ts)
        publish_metrics(metrics)
    driver.close()
