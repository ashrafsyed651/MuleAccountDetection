"""Scheduled Phase 3 graph analytics worker."""

import argparse
import os
import time
from pathlib import Path

from neo4j import GraphDatabase

from offline_tda_analysis import (
    calculate_topological_features,
    fetch_transaction_subgraph,
    save_persistence_diagram,
)


GRAPH_NAME = "aml_transaction_graph"


def wait_for_dependencies(driver) -> None:
    for attempt in range(1, 61):
        try:
            driver.verify_connectivity()
            with driver.session() as session:
                session.run("RETURN gds.version() AS version").single()
            print("Neo4j and GDS are ready")
            return
        except Exception as exc:
            if attempt == 60:
                raise RuntimeError("Neo4j GDS did not become ready") from exc
            print(f"Waiting for Neo4j GDS ({exc}); retrying in 2 seconds")
            time.sleep(2)


def run_gds_analytics(driver) -> None:
    """Refresh degree, SCC, and Louvain Account properties in Neo4j."""
    with driver.session() as session:
        exists = session.run(
            "CALL gds.graph.exists($graph_name) YIELD exists RETURN exists", graph_name=GRAPH_NAME
        ).single()["exists"]
        if exists:
            session.run("CALL gds.graph.drop($graph_name)", graph_name=GRAPH_NAME).consume()

        projection = session.run(
            """
            CALL gds.graph.project(
                $graph_name,
                'Account',
                {TRANSFER: {orientation: 'NATURAL'}}
            )
            YIELD nodeCount, relationshipCount
            RETURN nodeCount, relationshipCount
            """,
            graph_name=GRAPH_NAME,
        ).single()
        print(f"Projected {projection['nodeCount']} accounts and {projection['relationshipCount']} transfers")

        session.run(
            "CALL gds.degree.write($graph_name, {writeProperty: 'node_degree'})",
            graph_name=GRAPH_NAME,
        ).consume()
        session.run(
            "CALL gds.scc.write($graph_name, {writeProperty: 'scc_community_id'})",
            graph_name=GRAPH_NAME,
        ).consume()
        session.run(
            "CALL gds.louvain.write($graph_name, {writeProperty: 'louvain_community_id'})",
            graph_name=GRAPH_NAME,
        ).consume()
        session.run("CALL gds.graph.drop($graph_name)", graph_name=GRAPH_NAME).consume()


def write_tda_features(driver, features: dict[str, dict[str, float]]) -> None:
    rows = [
        {
            "account_id": account_id,
            "tda_cycle_count": int(values["tda_cycle_count"]),
            "tda_h1_persistence": float(values["tda_h1_persistence"]),
        }
        for account_id, values in features.items()
    ]
    with driver.session() as session:
        session.run(
            "MATCH (a:Account) SET a.tda_cycle_count = 0, a.tda_h1_persistence = 0.0"
        ).consume()
        if rows:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Account {account_id: row.account_id})
                SET a.tda_cycle_count = row.tda_cycle_count,
                    a.tda_h1_persistence = row.tda_h1_persistence
                """,
                rows=rows,
            ).consume()


def run_once(driver, limit: int, output_path: Path) -> None:
    run_gds_analytics(driver)
    graph = fetch_transaction_subgraph(driver, limit)
    features, diagrams = calculate_topological_features(graph)
    write_tda_features(driver, features)
    save_persistence_diagram(diagrams, output_path)
    print(f"Phase 3 refresh complete for {len(features)} transaction-connected accounts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh GDS and TDA account features")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://neo4j:7687"))
    parser.add_argument("--username", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "password"))
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", default="/artifacts/tda_persistence_diagram.png")
    args = parser.parse_args()

    with GraphDatabase.driver(args.uri, auth=(args.username, args.password)) as driver:
        wait_for_dependencies(driver)
        while True:
            try:
                run_once(driver, args.limit, Path(args.output))
            except Exception as exc:
                print(f"Phase 3 refresh failed: {exc}")
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
