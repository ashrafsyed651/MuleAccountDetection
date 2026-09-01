# models/gnn_embedder.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import os

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        # Return 64-dimensional node embeddings
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels=64):
        super(LinkPredictor, self).__init__()
        self.lin = torch.nn.Linear(in_channels, 1)

    def forward(self, z, edge_label_index):
        # Compute dot product or simple neural network on concatenated embeddings
        src_embeddings = z[edge_label_index[0]]
        dst_embeddings = z[edge_label_index[1]]
        # Hadamard product
        h = src_embeddings * dst_embeddings
        return self.lin(h).squeeze(-1)

def train_unsupervised():
    """Example training loop for unsupervised link prediction."""
    # Dummy data for demonstration purposes
    num_nodes = 1000
    num_node_features = 10
    x = torch.randn((num_nodes, num_node_features))
    edge_index = torch.randint(0, num_nodes, (2, 5000))
    
    data = Data(x=x, edge_index=edge_index)
    
    # Split edges for link prediction
    transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=False, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(data)
    
    model = GraphSAGE(in_channels=num_node_features, out_channels=64)
    predictor = LinkPredictor(in_channels=64)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    predictor.train()
    
    for epoch in range(1, 51):
        optimizer.zero_grad()
        
        # Get embeddings
        z = model(train_data.x, train_data.edge_index)
        
        # Predict links (positive + negative)
        out = predictor(z, train_data.edge_label_index)
        loss = criterion(out, train_data.edge_label.float())
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

    print("Training complete. Saving TorchScript model...")
    # Export for Triton
    # We trace only the GraphSAGE model since we only need embeddings in production
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (train_data.x, train_data.edge_index))
    
    os.makedirs("triton_model_repository/gnn_embedder/1", exist_ok=True)
    traced_model.save("triton_model_repository/gnn_embedder/1/model.pt")
    print("Saved traced GraphSAGE to triton_model_repository/gnn_embedder/1/model.pt")

if __name__ == "__main__":
    train_unsupervised()
