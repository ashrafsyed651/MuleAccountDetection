# tests/test_phase3.py
"""
Phase 3 End-to-End Test Suite
=============================
1. Seeds a synthetic A->B->C->D->A cycle (4 nodes) in Neo4j.
   A 4-node ring creates a genuine H1 hole in persistent homology (a 3-node
   triangle is a filled 2-simplex, so ripser correctly reports no H1 bar).
2. Verifies the TDA engine detects the cycle (Betti-1 > 0, persistence > 0).
3. Cleans up after itself.

Run from the project root:
    python -m pytest tests/test_phase3.py -v
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is on path so we can import graph_analytics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Import the TDA engine functions we want to test
# ---------------------------------------------------------------------------
from graph_analytics.tda_engine import (
    compute_tda_features,
    fetch_suspect_subgraph,
)

# ---------------------------------------------------------------------------
# Neo4j connection (host-side: localhost:7687)
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


@pytest.fixture(scope="module")
def driver():
    """Create a Neo4j driver that is shared across all tests in this module."""
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    drv.verify_connectivity()
    yield drv
    drv.close()


# ---------------------------------------------------------------------------
# Helpers: seed and teardown synthetic data
# ---------------------------------------------------------------------------

# Use a 4-node ring so that ripser can detect a genuine topological hole (H1).
# A 3-node triangle is a filled 2-simplex with Betti-1 = 0, which is correct
# mathematically but not useful for our test.
TEST_ACCOUNTS = ["TestMule_A", "TestMule_B", "TestMule_C", "TestMule_D"]


def _seed_cycle(driver):
    """Insert a directed 4-ring: A->B->C->D->A with known amounts."""
    query = """
    MERGE (a:Account {account_id: 'TestMule_A'})
    MERGE (b:Account {account_id: 'TestMule_B'})
    MERGE (c:Account {account_id: 'TestMule_C'})
    MERGE (d:Account {account_id: 'TestMule_D'})
    MERGE (a)-[:TRANSFER {amount: 5000.0, timestamp: timestamp()}]->(b)
    MERGE (b)-[:TRANSFER {amount: 5000.0, timestamp: timestamp()}]->(c)
    MERGE (c)-[:TRANSFER {amount: 5000.0, timestamp: timestamp()}]->(d)
    MERGE (d)-[:TRANSFER {amount: 5000.0, timestamp: timestamp()}]->(a)
    """
    with driver.session() as session:
        session.run(query)


def _teardown_cycle(driver):
    """Remove the synthetic test nodes and relationships."""
    query = """
    MATCH (n:Account)
    WHERE n.account_id IN $ids
    DETACH DELETE n
    """
    with driver.session() as session:
        session.run(query, ids=TEST_ACCOUNTS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTDAEngine:
    """Validate that the TDA engine correctly identifies a synthetic cycle."""

    @pytest.fixture(autouse=True)
    def _setup_teardown(self, driver):
        """Seed data before, clean up after each test class run."""
        _seed_cycle(driver)
        yield
        _teardown_cycle(driver)

    def test_subgraph_contains_test_nodes(self, driver):
        """After seeding, the subgraph must include our four test accounts."""
        graph = fetch_suspect_subgraph(driver, limit=5000)
        for acct in TEST_ACCOUNTS:
            assert graph.has_node(acct), f"Node '{acct}' missing from subgraph"

    def test_subgraph_has_cycle_edges(self, driver):
        """The directed edges A->B, B->C, C->D, D->A must exist."""
        graph = fetch_suspect_subgraph(driver, limit=5000)
        assert graph.has_edge("TestMule_A", "TestMule_B")
        assert graph.has_edge("TestMule_B", "TestMule_C")
        assert graph.has_edge("TestMule_C", "TestMule_D")
        assert graph.has_edge("TestMule_D", "TestMule_A")

    def test_tda_detects_cycle(self, driver):
        """The TDA engine must detect at least one cycle (tda_cycle_count > 0)."""
        graph = fetch_suspect_subgraph(driver, limit=5000)
        features = compute_tda_features(graph)

        for acct in TEST_ACCOUNTS:
            assert acct in features, f"Features missing for '{acct}'"
            assert features[acct]["tda_cycle_count"] > 0, (
                f"Account '{acct}' was expected to participate in a cycle "
                f"but tda_cycle_count = {features[acct]['tda_cycle_count']}"
            )

    def test_tda_h1_persistence_positive(self, driver):
        """Betti-1 persistence for cycle nodes must be > 0 (genuine topological hole)."""
        graph = fetch_suspect_subgraph(driver, limit=5000)
        features = compute_tda_features(graph)

        for acct in TEST_ACCOUNTS:
            assert features[acct]["tda_h1_persistence"] > 0.0, (
                f"Account '{acct}' persistence = {features[acct]['tda_h1_persistence']}; "
                f"expected > 0 for a node on a 4-ring cycle"
            )


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main(["-v", __file__])
