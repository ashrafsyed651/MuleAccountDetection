import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            feast_features = pb_utils.get_input_tensor_by_name(request, "FEAST_FEATURES").as_numpy()
            gnn_embedding = pb_utils.get_input_tensor_by_name(request, "GNN_EMBEDDING").as_numpy()
            
            # gnn_embedding might be shape [N, 64] where N is number of nodes in subgraph.
            # We assume the first node is the target node for which we want the prediction.
            if len(gnn_embedding.shape) > 1:
                target_embedding = gnn_embedding[0:1, :] # shape [1, 64]
            else:
                target_embedding = np.expand_dims(gnn_embedding, 0)
                
            # feast_features shape [1, 6]
            # Concatenate along axis=1 -> shape [1, 70]
            combined = np.concatenate([feast_features, target_embedding], axis=1)
            
            out_tensor = pb_utils.Tensor("COMBINED_FEATURES", combined)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            
        return responses
