import pickle


def save_model_weights(client_id, weights):
    # Serialize the weights list using pickle
    filename = f"./FL_WS_BRAIN_TUMOR v2/server/received_weights/Client_{client_id}.weights.h5"  # Use `.pkl` for serialization
    # Save weights to an HDF5 file
    with open(filename, 'wb') as file:
        pickle.dump(weights, file)
    print(f"Saved weights as Client_{client_id}.weights.h5")


def save_trained_model(global_model,average_weights):
    
    if average_weights is None:
        print("No weights to aggregate. Global model not updated.")
        return
    
    # Set the weights to the global model
    global_model.set_weights(average_weights)
    print("Averaged weights assigned to the global model.")
    
    # Save the global model
    global_model.save('./FL_WS_BRAIN_TUMOR v2/server/Trained_global_model.keras')
    print("Global model saved as Trained_global_model.keras")
