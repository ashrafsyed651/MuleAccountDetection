# tests/test_phase5.py
"""
Phase 5 Test Suite
==================
Tests Explainability & Analyst Feedback Loop:
1. Explainability Engine (SHAP + GNNExplainer -> ExplanationReport).
2. Analyst Feedback FastAPI endpoints (GET explanation & POST disposition).
3. Incremental XGBoost retraining job.

Run from project root:
    python -m pytest tests/test_phase5.py -v
"""

import os
import sys
import json
import pytest
import numpy as np
import xgboost as xgb
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from explainability.explainer import HybridExplainer, XGBoostExplainer, GNNSubGraphExplainer
from api.feedback import app
from mlops.retrain import incremental_retrain


# ---------------------------------------------------------------------------
# Test 1: Explainability Engine
# ---------------------------------------------------------------------------

class TestExplainabilityEngine:

    @pytest.fixture
    def mock_xgb_model(self):
        """Create a fitted XGBoost model for SHAP testing."""
        np.random.seed(42)
        X = np.random.randn(100, 70).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, eval_metric="logloss")
        model.fit(X, y)
        return model

    def test_xgb_explainer(self, mock_xgb_model):
        explainer = XGBoostExplainer(mock_xgb_model)
        sample_vector = np.random.randn(70).astype(np.float32)
        top_drivers = explainer.explain_instance(sample_vector, top_k=5)

        assert len(top_drivers) == 5, f"Expected 5 top drivers, got {len(top_drivers)}"
        for item in top_drivers:
            assert "feature" in item
            assert "shap_value" in item
            assert "impact_percentage" in item

    def test_gnn_subgraph_explainer(self):
        explainer = GNNSubGraphExplainer()
        subgraph = explainer.explain_node("Acc100", top_k_edges=5)

        assert subgraph["target_account"] == "Acc100"
        assert subgraph["hop_depth"] == 2
        assert len(subgraph["influential_edges"]) == 5

    def test_hybrid_explainer_report(self, mock_xgb_model):
        explainer = HybridExplainer(xgb_model=mock_xgb_model)
        sample_vector = np.random.randn(70).astype(np.float32)

        report = explainer.generate_report(
            alert_id="ALT-999",
            account_id="Acc1001",
            fraud_probability=0.88,
            feature_vector=sample_vector
        )

        assert report["alert_id"] == "ALT-999"
        assert report["account_id"] == "Acc1001"
        assert report["fraud_probability"] == 0.88
        assert report["risk_level"] == "HIGH"
        assert len(report["shap_top_drivers"]) == 5
        assert report["gnn_subgraph_explanation"]["hop_depth"] == 2


# ---------------------------------------------------------------------------
# Test 2: Analyst Feedback FastAPI Endpoints
# ---------------------------------------------------------------------------

class TestFeedbackAPI:

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "HEALTHY"

    def test_get_explanation_report(self, client):
        response = client.get("/api/alerts/ALT-123/explanation?account_id=Acc99&fraud_probability=0.92")
        assert response.status_code == 200
        data = response.json()
        assert data["alert_id"] == "ALT-123"
        assert data["risk_level"] == "HIGH"
        assert "shap_top_drivers" in data

    def test_submit_disposition_true_positive(self, client):
        payload = {
            "status": "TRUE_POSITIVE",
            "analyst_notes": "Confirmed circular transaction pattern",
            "account_id": "Acc99"
        }
        response = client.post("/api/alerts/ALT-123/disposition", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["alert_id"] == "ALT-123"
        assert data["status"] == "TRUE_POSITIVE"
        assert data["topic"] == "analyst-feedback"

    def test_submit_disposition_false_positive(self, client):
        payload = {
            "status": "FALSE_POSITIVE",
            "analyst_notes": "Legitimate business account",
            "account_id": "Acc100"
        }
        response = client.post("/api/alerts/ALT-124/disposition", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FALSE_POSITIVE"


# ---------------------------------------------------------------------------
# Test 3: Incremental Retraining Job
# ---------------------------------------------------------------------------

class TestRetrainingJob:

    def test_incremental_retrain(self, tmp_path):
        model_path = os.path.join(str(tmp_path), "xgboost.json")
        
        # Initial train & save
        X_init = np.random.randn(50, 70).astype(np.float32)
        y_init = np.random.randint(0, 2, 50)
        dtrain = xgb.DMatrix(X_init, label=y_init)
        booster = xgb.train({"objective": "binary:logistic"}, dtrain, num_boost_round=5)
        booster.save_model(model_path)

        # Mock feedback batch
        batch = [
            {"alert_id": f"ALT-{i}", "label": 1 if i % 2 == 0 else 0, "feature_vector": np.random.randn(70).tolist()}
            for i in range(10)
        ]

        # Perform incremental retrain
        success = incremental_retrain(batch, model_path=model_path)
        assert success is True
        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
