
import json
import torch


'''
Input: file such that each line is a JSON object.
Output: A JSON file containing an array, where each element is a JSON object.
'''
def preprocess_max_layer(input_file_name='counterfact_return_mlp.json', output_file_name='counterfact_max_layer_requests.json'):
    processed_data = [] 
    seen = set()
    with open(input_file_name, 'r') as input_file:
        for line in input_file:
            # Directly parse each line as a JSON object
            data = json.loads(line.strip())
            case_id = data["case_id"]

            # Remove duplicate cases
            if case_id in seen:
                continue
            seen.add(case_id)
            # scores = torch.Tensor(data["scores"])
            # end_idx = data["subject_range"][1]
            # layers = [torch.argmax(scores[end_idx]).item()]
            # data["max_score_layer"] = layers
            # data.pop('correct_prediction', None) 
            # data.pop('scores', None)
            processed_data.append(data)

    with open(output_file_name, 'w') as output_file:
        json.dump(processed_data, output_file, indent=2)

if __name__ == "__main__":
    preprocess_max_layer('zsre_comb.json', 'zsre_comb_max_layer_requests.json')