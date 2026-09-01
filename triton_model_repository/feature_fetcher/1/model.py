import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(self.model_config, "FEAST_FEATURES")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        responses = []
        for request in requests:
            account_ids = pb_utils.get_input_tensor_by_name(request, "ACCOUNT_ID").as_numpy()
            
            # Mocking Feast feature retrieval: [transaction_volume_30d, node_degree, scc_community_id, louvain_community_id, tda_cycle_count, tda_h1_persistence]
            # In a real setup, we would use Feast online store here.
            features = np.random.randn(account_ids.shape[0], 6).astype(self.output_dtype)
            
            out_tensor = pb_utils.Tensor("FEAST_FEATURES", features)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            
        return responses
