"""Register the Neo4j CDC source connector once Kafka Connect is healthy.

Kafka Connect's PUT endpoint is idempotent, so this can safely run on every
`docker compose up` and keeps the connector definition in version control.
"""

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CONNECT_URL = "http://kafka-connect:8083"
CONNECTOR_NAME = "neo4j-cdc-source"
CONFIG_PATH = Path("/config/neo4j-connector.json")


def register_connector() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["config"]
    request = Request(
        f"{CONNECT_URL}/connectors/{CONNECTOR_NAME}/config",
        data=json.dumps(config).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="PUT",
    )

    for attempt in range(1, 31):
        try:
            with urlopen(request, timeout=10) as response:
                print(f"Registered {CONNECTOR_NAME}: HTTP {response.status}")
                return
        except (HTTPError, URLError) as exc:
            if attempt == 30:
                raise RuntimeError(f"Could not register {CONNECTOR_NAME}") from exc
            print(f"Kafka Connect is not ready ({exc}); retrying in 2 seconds")
            time.sleep(2)


if __name__ == "__main__":
    register_connector()
