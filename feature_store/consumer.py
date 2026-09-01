import json
import time
from datetime import datetime, timezone

import pandas as pd
from feast import FeatureStore
from kafka import KafkaConsumer


def parse_cdc_event(raw_event):
    """Parse a Kafka Connect envelope or a legacy schemaless event."""
    event = raw_event
    for _ in range(3):
        if isinstance(event, str):
            event = json.loads(event)
        else:
            break
    # JsonConverter with schemas enabled wraps the actual CDC event in payload.
    if isinstance(event, dict) and "schema" in event and isinstance(event.get("payload"), dict):
        event = event["payload"]
    return event if isinstance(event, dict) else None


def extract_account_features(event):
    """Extract features from native Neo4j CDC, old CDC, or flat test events."""
    # Native Neo4j CDC: {"event": {"eventType": "n", "labels": [...],
    # "state": {"after": {"properties": {...}}}}}. Deletes have after=None.
    cdc_event = event.get("event")
    if isinstance(cdc_event, dict):
        if cdc_event.get("eventType") != "n" or "Account" not in cdc_event.get("labels", []):
            return None, None
        after = cdc_event.get("state", {}).get("after")
        if not isinstance(after, dict):
            return None, None
        properties = after.get("properties", after)
        if isinstance(properties, dict) and properties.get("account_id") is not None:
            return str(properties["account_id"]), properties

    # Compatibility with the short-lived pre-5.x CDC envelope.
    payload = event.get("payload")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        after = payload.get("after", {})
        if isinstance(after, str):
            after = json.loads(after)
        labels = after.get("labels", []) if isinstance(after, dict) else []
        if "Account" in labels:
            properties = after.get("properties", after)
            if isinstance(properties, str):
                properties = json.loads(properties)
            account_id = properties.get("account_id") if isinstance(properties, dict) else None
            if account_id:
                return str(account_id), properties

    # Flat polling connector: {"account_id": "AccA", "transaction_volume_30d": 50000.0, ...}
    account_id = event.get("account_id")
    if account_id is not None:
        return str(account_id), event

    return None, None


def run_consumer():
    print("Starting Feast CDC Consumer...")
    time.sleep(5)

    store = FeatureStore(repo_path=".")
    consumer = KafkaConsumer(
        "graph-mutations-log",
        bootstrap_servers=["kafka:29092"],
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="feast-cdc-group-v3",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    print("Listening for Neo4j CDC messages on 'graph-mutations-log'...")
    for message in consumer:
        raw_event = message.value
        print(f"Received CDC event: {raw_event}")

        try:
            event = parse_cdc_event(raw_event)
            if not event:
                print("Skipping event: could not parse to dict")
                continue

            account_id, properties = extract_account_features(event)
            if not account_id:
                print("Skipping non-Account event (no account_id found)")
                continue

            feature_data = {
                "account_id": [account_id],
                "transaction_volume_30d": [float(properties.get("transaction_volume_30d") or 0.0)],
                "node_degree": [int(properties.get("node_degree") or 0)],
                "scc_community_id": [int(properties.get("scc_community_id") or 0)],
                "louvain_community_id": [int(properties.get("louvain_community_id") or 0)],
                "tda_cycle_count": [int(properties.get("tda_cycle_count") or 0)],
                "tda_h1_persistence": [float(properties.get("tda_h1_persistence") or 0.0)],
                "event_timestamp": [datetime.now(timezone.utc)],
            }
            df = pd.DataFrame(feature_data)
            print(f"Pushing Account features to Feast: {feature_data}")
            store.push("graph_mutations_push_source", df)
            print(f"Successfully pushed features for account: {account_id}")

        except Exception as e:
            print(f"Error parsing/pushing CDC event: {e}")


if __name__ == "__main__":
    run_consumer()
