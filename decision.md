# Architectural Decisions

This document tracks all meaningful architectural and technical decisions made during the development of the high-velocity data ingestion pipeline, graph analytics engine, and hybrid ML inference pipeline (Phases 1–4).

## 1. Apache Kafka in KRaft Mode
*Decision:* We are deploying Apache Kafka using KRaft (Kafka Raft metadata) mode instead of Zookeeper.
*Reasoning:* KRaft is the modern consensus protocol for Kafka, removing the dependency on Zookeeper. This simplifies the infrastructure, reduces the operational burden of managing two distributed systems, and improves scalability by allowing Kafka brokers to manage metadata internally.

## 2. Apache Flink with PyFlink
*Decision:* We are using Apache Flink for stream processing and implementing the logic using PyFlink (Python API).
*Reasoning:* 
- Flink provides robust, exactly-once processing semantics and advanced state management (like sliding windows), which are essential for calculating precise real-time velocity metrics.
- Python was chosen over Java/Scala because the subsequent phases of the pipeline will involve Graph Neural Networks (GNNs) and Topological Data Analysis (TDA), which are heavily reliant on Python ecosystems (e.g., PyTorch Geometric, NetworkX). Using PyFlink allows for seamless ML integration natively in Python without complex language boundaries.

## 3. Mutual TLS (mTLS) for Security
*Decision:* Strict mTLS is enforced for communication between Kafka brokers and Flink workers.
*Reasoning:* Security is a primary mandate (zero-trust). By using mTLS, we ensure that:
- Data in transit is encrypted.
- Only authenticated and authorized clients (possessing the correct signed certificate) can connect to the Kafka cluster, preventing unauthorized data injection or extraction.

## 4. Containerized Local Development
*Decision:* The local infrastructure is managed via Docker Compose.
*Reasoning:* Provides a reproducible, isolated environment that mimics the production containerized deployment (e.g., Kubernetes). Certificates are generated locally and injected via Docker volume mounts.

## 5. Neo4j Enterprise as the Graph Database (Phase 2)
*Decision:* Selected Neo4j Enterprise Edition over Community Edition/TigerGraph/NebulaGraph.
*Reasoning:* Neo4j has massive adoption in the financial sector for forensics and robust Cypher pattern matching. More importantly, native CDC is strictly an Enterprise Edition feature (Community Edition ignores the CDC flag), which is required to tap the transaction WAL.

## 6. Official Neo4j 5.x Kafka Connector for WAL CDC (Phase 2)
*Decision:* Configured the official Neo4j Kafka Connector source using the WAL-based Neo4jConnector class instead of legacy query polling.
*Reasoning:* Legacy stream connectors use polling queries (MATCH ...) which create heavy query load on the database. Using the native WAL-based org.neo4j.connectors.kafka.source.Neo4jConnector with pattern-matching ensures true CDC with zero-overhead streaming of node mutations.

## 7. Feast Feature Store with Redis & Python Ingestion Worker (Phase 2)
*Decision:* Configured Feast with a PushSource and an active Python Kafka consumer instead of a Spark-based streaming worker.
*Reasoning:* Stream feature views in Feast with mode="spark" require PySpark/JVM environments, which are too heavy for basic Python containers. Transitioning to a PushSource and writing a custom, lightweight Python Kafka consumer avoids these heavy runtimes while ensuring transaction nodes' CDC mutations are cleanly pushed into Redis in real-time.

## 8. Dual-Track Graph Analytics Engine: Real-Time Heuristics & Offline TDA (Phase 3)
*Decision:* Split graph feature generation into two parallel tracks: Neo4j Graph Data Science (GDS) for real-time community/structure heuristics, and an offline Topological Data Analysis (TDA) engine using `ripser` for persistent homology calculations.
*Reasoning:* 
- **Real-time Fast Heuristics:** Tarjan's Strongly Connected Components (SCC) and Louvain community detection run efficiently inside Neo4j GDS over a rolling 24-hour window, capturing localized cluster density.
- **Offline TDA Engine:** Circular money layering (e.g., $A \rightarrow B \rightarrow C \rightarrow D \rightarrow A$) can be obfuscated across long time windows and intermediate accounts. Computing 1D persistent homology ($H_1$ Betti numbers) using inverse transfer volume as distance mathematically isolates cyclic flows as non-trivial topological holes.
- **Cycle Sizing & Filtration:** A 3-node triangle acts as a filled 2-simplex with $H_1 = 0$, whereas a 4+-node loop produces a non-zero $H_1$ persistence bar, accurately signaling complex circular layering schemes.

## 9. Hybrid GNN + XGBoost Architecture with NVIDIA Triton Ensemble (Phase 4)
*Decision:* Implemented a 2-stage hybrid model architecture (PyTorch Geometric GraphSAGE + `xgboost.XGBClassifier`) served via an NVIDIA Triton Ensemble Pipeline.
*Reasoning:* 
- **GraphSAGE Embedder:** Graph Neural Networks excel at learning local structural representations ($64$-dim embeddings) via message passing across node neighborhoods.
- **XGBoost Classifier:** Gradient Boosted Decision Trees outperform raw GNNs on heterogeneous tabular features (transaction velocity, Betti-1 persistence, degree). Concatenating GNN embeddings with tabular features gives XGBoost complete contextual and structural visibility.
- **Imbalance Handling:** Mule accounts are extremely rare ($< 5\%$ of total accounts). Setting `scale_pos_weight = N_neg / N_pos` in `XGBClassifier` prevents majority-class bias.
- **Triton Ensemble DAG:** NVIDIA Triton handles high-throughput serving by orchestrating a 4-step pipeline (`feature_fetcher` $\rightarrow$ `gnn_embedder` $\rightarrow$ `feature_combiner` $\rightarrow$ `xgb_classifier` FIL) in a single gRPC endpoint call, eliminating inter-service HTTP serialization bottlenecks.

## 10. Explainability & Human-in-the-Loop Active Learning (Phase 5)
*Decision:* Implemented a dual SHAP + GNNExplainer engine wrapped in FastAPI, paired with a Kafka-driven active learning retraining consumer.
*Reasoning:* 
- **SHAP TreeExplainer & GNNExplainer:** Fraud analysts require mathematical rationale for flagged alerts. SHAP quantifies top tabular/topological drivers (e.g., $+30\%$ risk from Betti-1 loop count), while GNNExplainer isolates the key 2-hop transaction neighborhood.
- **Kafka-Driven Disposition Stream:** Analyst verdicts (`TRUE_POSITIVE` / `FALSE_POSITIVE`) are published to the `analyst-feedback` Kafka topic to ensure decoupling between web API and background MLOps.
- **Continuous Incremental Retraining:** The `retrain.py` consumer collects ground truth dispositions and invokes `xgb.train(..., xgb_model=existing_model)` every 100 dispositions, updating model weights incrementally and triggering Triton model reloads via REST API without downtime.
