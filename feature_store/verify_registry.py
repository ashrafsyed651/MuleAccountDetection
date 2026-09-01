"""Fail fast if Feast did not register the Phase 3 feature view."""

from feast import FeatureStore


if __name__ == "__main__":
    store = FeatureStore(repo_path=".")
    feature_view = store.get_feature_view("account_graph_features")
    if feature_view.source.name != "graph_mutations_push_source":
        raise RuntimeError("account_graph_features is not backed by graph_mutations_push_source")
    print("Feast registry contains account_graph_features and its PushSource")
