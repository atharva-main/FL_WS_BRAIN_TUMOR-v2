import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import asyncio
import pickle
import types
import websockets
from cryptography.fernet import Fernet

CLIENT_ID = 10
data_set = "./FL_WS_BRAIN_TUMOR v2/clients/C1"
SERVER_URI = 'ws://127.0.0.1:8000'

def key_checking():
    k = input("Enter the key: ")
    try:
        fernet_key = k.encode()  
        cipher = Fernet(fernet_key)
        print("Key is valid and cipher has been created.")
        return cipher
    except Exception as e:
        print(f"Error initializing Fernet cipher: {e}")
        return None

cipher = key_checking()

def encrypt_obj(obj):
    return cipher.encrypt(obj)

def decrypt_obj(encrypted_data):
    return cipher.decrypt(encrypted_data)

async def receive_data_from_server(websocket):
    try:
        # Receive and decrypt the model
        model_size = int(await websocket.recv())
        print(f"Expected model size: {model_size} bytes")

        encrypted_model = bytearray()
        while len(encrypted_model) < model_size:
            chunk = await websocket.recv()
            encrypted_model.extend(chunk)
            # print(f"Received chunk of size: {len(chunk)} bytes")

        # Check type before decrypting
        if isinstance(encrypted_model, (bytes, bytearray)):
            decrypted_model = decrypt_obj(bytes(encrypted_model))
            local_model = pickle.loads(decrypted_model)
            print("Received and decrypted the model from server.")
        else:
            print("Error: Encrypted model data is not bytes.")

        # Receive and decrypt the .py file
        file_size = int(await websocket.recv())
        print(f"Expected file size: {file_size} bytes")

        encrypted_file = bytearray()
        while len(encrypted_file) < file_size:
            chunk = await websocket.recv()
            encrypted_file.extend(chunk)

        # Check type before decrypting
        if isinstance(encrypted_file, (bytes, bytearray)):
            decrypted_file = decrypt_obj(bytes(encrypted_file))
            print("Received and decrypted the file content from server.")
        else:
            print("Error: Encrypted file data is not bytes.")
        
        return local_model, decrypted_file
        
    except Exception as e:
        print(f'Error during data reception: {e}')


async def send_data_to_server(model_weights, websocket):
    try:
        # Send client ID 
        await websocket.send(str(CLIENT_ID))

        # Serialize and encrypt the model weights to send back
        serialize_model_weights = pickle.dumps(model_weights)
        encrypted_model_weights = encrypt_obj(serialize_model_weights)
        
        await websocket.send(str(len(encrypted_model_weights)).encode())
        chunk_size = 1024
        for i in range(0, len(encrypted_model_weights),chunk_size):
            await websocket.send(encrypted_model_weights[i:i+ chunk_size])

        #await websocket.send(encrypted_model_weights)
        print(f"Client {CLIENT_ID} sent client ID and model weights to server.")

    except Exception as e:
        print(f'Error during data sending: {e}')

def handling_training(local_model, decrypted_file):
    try:
        # Creating a module from the decrypted file content without saving it to disk
        client_functions = types.ModuleType("client_functions")
        
        # Execute the decrypted file content in the module's namespace
        exec(decrypted_file.decode(), client_functions.__dict__)

        # Check if the required functions exist in the decrypted module
        if not hasattr(client_functions, "data_processing") or not hasattr(client_functions, "train_model"):
            raise AttributeError("Decrypted file does not contain required functions.")

        # Use functions from the in-memory module for data processing and training
        train_gen, val_gen = client_functions.data_processing(data_set)
        client_functions.train_model(local_model, train_gen, val_gen)
        model_weights = local_model.get_weights()
        print("Model trained successfully.")

        return model_weights

    except Exception as e:
        print(f'Error during model training: {e}')

async def start_client():
    try:
        async with websockets.connect(SERVER_URI) as websocket:
            print(f"Connected to server at {SERVER_URI}")

            # Get local_model and file from server
            local_model, decrypted_file = await receive_data_from_server(websocket)

            # Train model and get weights
            print('Model training starting')
            model_weights = handling_training(local_model, decrypted_file)

            # Send weights to the server
            await send_data_to_server(model_weights, websocket)

    except Exception as e:
        print(f"Error in client-server communication: {e}")

if __name__ == "__main__":
    asyncio.run(start_client())
