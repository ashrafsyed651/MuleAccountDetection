"""Create the offline parquet file required by Feast FileSource on first boot."""
import os
from datetime import datetime, timezone

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PARQUET_PATH = os.path.join(DATA_DIR, "offline_store.parquet")


def bootstrap_offline_store():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(PARQUET_PATH):
        print(f"Offline store already exists: {PARQUET_PATH}")
        return

    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        {
            "account_id": ["bootstrap"],
            "transaction_volume_30d": [0.0],
            "node_degree": [0],
            "event_timestamp": [now],
            "created_timestamp": [now],
        }
    )
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Created bootstrap offline store: {PARQUET_PATH}")


if __name__ == "__main__":
    bootstrap_offline_store()
