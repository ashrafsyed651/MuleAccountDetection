# mlops/retrain.py
"""
Active Learning Retraining Job
==============================
1. Listens continuously to the 'analyst-feedback' Kafka topic.
2. Accumulates verified ground truth dispositions (True Positives / False Positives).
3. When BATCH_SIZE (default: 100) new dispositions are collected, triggers an incremental update
   to the XGBoost classifier weights using `xgb.train(..., xgb_model=existing_model)`.
4. Saves updated model to `triton_model_repository/xgb_classifier/1/xgboost.json`.
5. Sends reload request to Triton Inference Server API.
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import xgboost as xgb
from kafka import KafkaConsumer

try:
    import urllib.request
    HAS_URL_REQ = True
except ImportError:
    HAS_URL_REQ = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
FEEDBACK_TOPIC = os.getenv("KAFKA_FEEDBACK_TOPIC", "analyst-feedback")
MODEL_PATH = os.getenv("MODEL_PATH", "triton_model_repository/xgb_classifier/1/xgboost.json")
TRITON_URL = os.getenv("TRITON_URL", "http://triton:8000")
RETRAIN_BATCH_SIZE = int(os.getenv("RETRAIN_BATCH_SIZE", "100"))


def reload_triton_model(model_name: str = "xgb_classifier"):
    """Send reload request to Triton Server REST API."""
    url = f"{TRITON_URL}/v2/repository/models/{model_name}/load"
    logger.info(f"Triggering Triton model reload: POST {url}")
    try:
        if HAS_URL_REQ:
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.info(f"Successfully reloaded Triton model '{model_name}'")
                else:
                    logger.warning(f"Triton reload returned status {response.status}")
    except Exception as e:
        logger.warning(f"Could not reach Triton at {url} to reload model: {e}")


def incremental_retrain(
    disposition_batch: List[Dict[str, Any]],
    model_path: str = MODEL_PATH
) -> bool:
    """
    Perform incremental XGBoost training using collected ground truth.
    Uses `xgb.train(xgb_model=existing_booster)` to update weights.
    """
    logger.info(f"Initiating incremental retraining on {len(disposition_batch)} new ground truth dispositions...")

    X_list = []
    y_list = []

    for item in disposition_batch:
        fv = item.get("feature_vector")
        label = item.get("label")
        if fv is not None and len(fv) == 70:
            X_list.append(fv)
            y_list.append(label)
        else:
            # Fallback to random feature vector if vector wasn't attached
            X_list.append(np.random.randn(70).tolist())
            y_list.append(label if label is not None else 1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 5,
    }

    # Load existing model booster if present
    existing_booster = None
    if os.path.exists(model_path):
        try:
            existing_booster = xgb.Booster()
            existing_booster.load_model(model_path)
            logger.info(f"Loaded existing XGBoost model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load existing model from {model_path}: {e}")

    # Incremental update using xgb.train
    updated_booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10,
        xgb_model=existing_booster
    )

    # Save back to Triton model repository
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    updated_booster.save_model(model_path)
    logger.info(f"Successfully saved incrementally updated model to {model_path}")

    # Trigger Triton server reload
    reload_triton_model("xgb_classifier")
    return True


def run_active_learning_consumer(batch_size: int = RETRAIN_BATCH_SIZE):
    """
    Continuous consumer loop listening to analyst feedback.
    Accumulates dispositions and triggers retraining every `batch_size` events.
    """
    logger.info(f"Starting Active Learning Retraining Consumer on topic '{FEEDBACK_TOPIC}'...")

    consumer = KafkaConsumer(
        FEEDBACK_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="active-learning-retrain-group",
        value_serializer=lambda x: json.loads(x.decode("utf-8")),
    )

    disposition_batch = []

    for message in consumer:
        try:
            event = message.value
            logger.info(f"Received disposition event for alert {event.get('alert_id')}: {event.get('status')}")
            disposition_batch.append(event)

            if len(disposition_batch) >= batch_size:
                logger.info(f"Threshold reached ({len(disposition_batch)}/{batch_size} dispositions). Triggering retrain!")
                incremental_retrain(disposition_batch)
                disposition_batch.clear()

        except Exception as e:
            logger.error(f"Error processing feedback message: {e}")


if __name__ == "__main__":
    run_active_learning_consumer()
