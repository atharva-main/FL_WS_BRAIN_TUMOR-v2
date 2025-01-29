import numpy as np
import os
import pickle

def aggregate_weights(path):
    weight_files = [f for f in os.listdir(path) if f.endswith('.h5')]  # Use `.pkl` for deserialization
    if not weight_files:
        print("No weight files found.")
        return None  # Return None if no weights are found
    
    all_weights = []
    for file in weight_files:
        file_path = os.path.join(path, file)
        # Load the weights from the file
        with open(file_path, 'rb') as weight_file:
            weights = pickle.load(weight_file)
            all_weights.append(weights)
    
    # Ensure that all_weights is a list of weight arrays
    average_weights = [np.mean(layer, axis=0) for layer in zip(*all_weights)]  

    # Save the aggregated weights separately
    with open('./FL_WS_BRAIN_TUMOR v2/server/aggregation/aggregated_weights.weights.h5', 'wb') as file:
        pickle.dump(average_weights, file)
    print("Aggregated weights saved as aggregated_weights.weights.h5")
    return average_weights
