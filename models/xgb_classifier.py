# models/xgb_classifier.py
import pandas as pd
import numpy as np
import xgboost as xgb
from feast import FeatureStore
import os
from datetime import datetime

def get_training_data():
    """Fetch historical features from Feast and concatenate GNN embeddings."""
    # Assuming Feast is running in the feature_store/ directory
    store = FeatureStore(repo_path="feature_store/")
    
    # In a real scenario, you'd query a true entity dataframe of labeled accounts
    entity_df = pd.DataFrame.from_dict({
        "account_id": ["Acc1", "Acc2", "Acc3", "Acc4", "Acc5"],
        "event_timestamp": [datetime.now()] * 5,
        "is_mule": [0, 0, 1, 0, 1]  # The labels
    })

    # Fetch features from Feast
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "account_features:transaction_volume_30d",
            "account_features:node_degree",
            "account_features:scc_community_id",
            "account_features:louvain_community_id",
            "account_features:tda_cycle_count",
            "account_features:tda_h1_persistence"
        ]
    ).to_df()
    
    # Mock GNN embeddings (normally you would query the GNN or load from an offline store)
    # 64-dimensional embeddings
    embeddings = pd.DataFrame(
        np.random.randn(len(training_df), 64), 
        columns=[f"gnn_embed_{i}" for i in range(64)]
    )
    embeddings["account_id"] = training_df["account_id"].values
    
    # Join features and embeddings
    final_df = training_df.merge(embeddings, on="account_id")
    return final_df

def train_xgboost():
    print("Fetching training data...")
    df = get_training_data()
    
    # Drop identifiers and timestamp for training
    X = df.drop(["account_id", "event_timestamp", "is_mule"], axis=1)
    y = df["is_mule"]
    
    # Calculate scale_pos_weight for imbalance
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = float(neg_count) / float(pos_count)
    print(f"Class distribution - Negatives: {neg_count}, Positives: {pos_count}")
    print(f"Setting scale_pos_weight to {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    print("Training XGBoost classifier...")
    model.fit(X, y)
    print("Training complete.")
    
    # Save for Triton FIL backend
    os.makedirs("triton_model_repository/xgb_classifier/1", exist_ok=True)
    model.save_model("triton_model_repository/xgb_classifier/1/xgboost.json")
    print("Saved XGBoost model to triton_model_repository/xgb_classifier/1/xgboost.json")

if __name__ == "__main__":
    train_xgboost()
