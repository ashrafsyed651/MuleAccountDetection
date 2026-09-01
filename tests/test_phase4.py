# tests/test_phase4.py
"""
Phase 4 Test Suite
==================
Tests the hybrid ML classification pipeline:
1. XGBoost classifier trains on synthetic data and produces valid predictions.
2. Triton config files are structurally valid.
3. (Optional) GNN embedder architecture test if torch_geometric is available.

Run from the project root:
    python -m pytest tests/test_phase4.py -v
"""

import json
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Test 1: XGBoost classifier trains and predicts on synthetic data
# ---------------------------------------------------------------------------

class TestXGBClassifier:
    """Test the XGBoost training pipeline with fully synthetic data (no Feast)."""

    def test_xgb_train_and_predict(self):
        """Train XGBoost on synthetic features and verify output shape."""
        import xgboost as xgb

        np.random.seed(42)
        n_samples = 200
        n_feast_features = 6   # transaction_volume, node_degree, scc, louvain, tda_cycle, tda_h1
        n_gnn_features = 64    # GNN embedding dim

        # Synthetic feature matrix: 6 Feast + 64 GNN = 70 features
        X = np.random.randn(n_samples, n_feast_features + n_gnn_features).astype(np.float32)
        # Imbalanced labels: ~10% mule accounts
        y = np.zeros(n_samples, dtype=int)
        y[:20] = 1
        np.random.shuffle(y)

        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos_weight = float(neg_count) / float(pos_count)

        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        model.fit(X, y)

        # Predict probabilities
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 2), f"Expected (5, 2), got {proba.shape}"

        # Predict labels
        preds = model.predict(X[:5])
        assert preds.shape == (5,), f"Expected (5,), got {preds.shape}"
        assert set(preds).issubset({0, 1}), "Predictions must be binary"

    def test_xgb_model_save_and_load(self, tmp_path):
        """Model can be saved in JSON format (for Triton FIL) and reloaded."""
        import xgboost as xgb

        np.random.seed(42)
        X = np.random.randn(100, 70).astype(np.float32)
        y = np.random.randint(0, 2, 100)

        model = xgb.XGBClassifier(n_estimators=10, max_depth=2, eval_metric="logloss")
        model.fit(X, y)

        model_path = os.path.join(str(tmp_path), "xgboost.json")
        model.save_model(model_path)
        assert os.path.exists(model_path), "Model file was not saved"
        assert os.path.getsize(model_path) > 0, "Model file is empty"

        # Reload and verify predictions match
        loaded = xgb.XGBClassifier()
        loaded.load_model(model_path)
        orig_preds = model.predict(X[:10])
        loaded_preds = loaded.predict(X[:10])
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_xgb_handles_imbalance(self):
        """With scale_pos_weight, the model should not predict all-zeros on imbalanced data."""
        import xgboost as xgb

        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 70).astype(np.float32)
        y = np.zeros(n, dtype=int)
        y[:25] = 1  # 5% positive rate

        # Deliberately make positive class have distinct feature distribution
        X[y == 1, :6] += 3.0  # shift Feast features for mule accounts

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            scale_pos_weight=float((n - 25)) / 25.0,
            eval_metric="logloss",
        )
        model.fit(X, y)
        preds = model.predict(X)

        # Should predict at least some positives
        assert preds.sum() > 0, "Model predicted all negatives — imbalance handling failed"


# ---------------------------------------------------------------------------
# Test 2: Triton config.pbtxt files exist and are structurally valid
# ---------------------------------------------------------------------------

class TestTritonConfigs:
    """Validate that the Triton model repository has the required structure."""

    TRITON_DIR = os.path.join(PROJECT_ROOT, "triton_model_repository")

    EXPECTED_MODELS = [
        "gnn_embedder",
        "xgb_classifier",
        "feature_fetcher",
        "feature_combiner",
        "ensemble_pipeline",
    ]

    def test_triton_directory_exists(self):
        assert os.path.isdir(self.TRITON_DIR), "triton_model_repository/ directory missing"

    @pytest.mark.parametrize("model_name", EXPECTED_MODELS)
    def test_model_dir_and_config_exist(self, model_name):
        model_dir = os.path.join(self.TRITON_DIR, model_name)
        assert os.path.isdir(model_dir), f"Model directory '{model_name}/' missing"

        config_path = os.path.join(model_dir, "config.pbtxt")
        assert os.path.isfile(config_path), f"config.pbtxt missing in '{model_name}/'"

    @pytest.mark.parametrize("model_name", EXPECTED_MODELS)
    def test_config_has_required_fields(self, model_name):
        """Each config.pbtxt must have 'name', 'input', and 'output' fields."""
        config_path = os.path.join(self.TRITON_DIR, model_name, "config.pbtxt")
        with open(config_path, "r") as f:
            content = f.read()

        assert f'name: "{model_name}"' in content, f"Config missing 'name: \"{model_name}\"'"
        assert "input" in content, f"Config missing 'input' section"
        assert "output" in content, f"Config missing 'output' section"

    def test_ensemble_has_steps(self):
        """The ensemble config must define an ensemble_scheduling with steps."""
        config_path = os.path.join(self.TRITON_DIR, "ensemble_pipeline", "config.pbtxt")
        with open(config_path, "r") as f:
            content = f.read()

        assert "ensemble_scheduling" in content, "Ensemble config missing 'ensemble_scheduling'"
        assert "step" in content, "Ensemble config missing 'step' definitions"
        # Should reference all sub-models
        for sub_model in ["feature_fetcher", "gnn_embedder", "feature_combiner", "xgb_classifier"]:
            assert sub_model in content, f"Ensemble config missing reference to '{sub_model}'"

    def test_python_backends_have_model_py(self):
        """Python backend models must have a 1/model.py file."""
        for model_name in ["feature_fetcher", "feature_combiner"]:
            model_py = os.path.join(self.TRITON_DIR, model_name, "1", "model.py")
            assert os.path.isfile(model_py), f"Missing {model_name}/1/model.py"


# ---------------------------------------------------------------------------
# Test 3: GNN embedder (only if torch_geometric is installed)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch_geometric
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


@pytest.mark.skipif(not HAS_TORCH_GEOMETRIC, reason="torch_geometric not installed")
class TestGNNEmbedder:
    """Test the GraphSAGE model architecture."""

    def test_graphsage_output_shape(self):
        from models.gnn_embedder import GraphSAGE

        model = GraphSAGE(in_channels=10, hidden_channels=128, out_channels=64)
        x = torch.randn(50, 10)
        edge_index = torch.randint(0, 50, (2, 200))

        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)

        assert out.shape == (50, 64), f"Expected (50, 64), got {out.shape}"

    def test_graphsage_torchscript_export(self, tmp_path):
        from models.gnn_embedder import GraphSAGE

        model = GraphSAGE(in_channels=10, out_channels=64)
        model.eval()

        x = torch.randn(20, 10)
        edge_index = torch.randint(0, 20, (2, 80))

        traced = torch.jit.trace(model, (x, edge_index))
        path = os.path.join(str(tmp_path), "model.pt")
        traced.save(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main(["-v", __file__])
