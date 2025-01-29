import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import asyncio
import pickle
import websockets
import os
from user_input.input_model import Input_Model
from aggregation.aggregation import aggregate_weights
from security.cryptography import key_generate, encrypt_obj, decrypt_obj
from utils.utils import save_model_weights, save_trained_model


HOST = '127.0.0.1'
PORT = 8000

NUM_CLIENTS = 3
client_counter = 0
client_counter_lock = asyncio.Lock()  # Lock for thread-safe counter updates

DT_FILE_PATH = "./FL_WS_BRAIN_TUMOR v2/server/user_input/data_processing_and_training.py"
RECEIVED_FILE_PATH = './FL_WS_BRAIN_TUMOR v2/server/received_weights'

# Initialize global model
global_model = Input_Model()
print("GLobal Model")
print(global_model.summary())
print('Global model initialize')

# Ensure the 'received_files' directory exists
if not os.path.exists("./FL_WS_BRAIN_TUMOR v2/server/received_weights"):
    os.makedirs("./FL_WS_BRAIN_TUMOR v2/server/received_weights")

#Key generation
cipher_suite = key_generate()

async def send_model(websocket, file):
    try:
        model = Input_Model()
        serialized_model = pickle.dumps(model)
        encrypted_model = encrypt_obj(cipher_suite,serialized_model)

        await websocket.send(str(len(encrypted_model)).encode())
        
        # Send in chunks
        chunk_size = 1024  # Adjust this size as needed
        for i in range(0, len(encrypted_model), chunk_size):
            await websocket.send(encrypted_model[i:i + chunk_size])

        print("Model sent successfully.")

        # Sending the file
        try:
            with open(file, 'rb') as f:
                file_data = f.read()
            encrypted_file = encrypt_obj(cipher_suite,file_data)

            await websocket.send(str(len(encrypted_file)).encode())

            # Send in chunks
            for i in range(0, len(encrypted_file), chunk_size):
                await websocket.send(encrypted_file[i:i + chunk_size])

            print("File sent successfully.")
        except Exception as e:
            print(f"Error during file sending: {e}")

    except Exception as e:
        print(f"Error during data sending: {e}")

async def receive_data_from_client(websocket):
    try:
        client_id = await websocket.recv()
        print(f"Received client ID: {client_id}")

        encrypted_weights_size = int(await websocket.recv())
        encrypted_weights = bytearray()

        while len(encrypted_weights) < encrypted_weights_size:
            encrypted_weights.extend(await websocket.recv())

        decrypted_weights = decrypt_obj(cipher_suite, bytes(encrypted_weights))
        client_model_weights = pickle.loads(decrypted_weights)
        print(f"Received and decrypted model weights from client {client_id}")

        async with client_counter_lock:  # Use async with here
            save_model_weights(client_id, client_model_weights)
            global client_counter
            client_counter += 1

        return client_id, client_model_weights

    except Exception as e:
        print(f"Error receiving data from client: {e}")
        return None, None


async def handle_client(websocket):
    try:
        await send_model(websocket, DT_FILE_PATH)
        client_id, received_weights = await receive_data_from_client(websocket)

        # if client_id and received_weights:
        #     save_model_weights(client_id, received_weights)

        if client_counter == NUM_CLIENTS:
            aw = aggregate_weights(RECEIVED_FILE_PATH)
            save_trained_model(global_model,aw)
            print("All clients' data received and aggregated. Server will stop now.")
            await stop_server()
    except Exception as e:
        print(f"Error in handling client: {e}")
    
async def stop_server():
    print("Stopping server...")
    # Cancel all running tasks except for this one
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    # Allow tasks to cancel gracefully
    await asyncio.gather(*tasks, return_exceptions=True)
    print("Server stopped gracefully.")

async def start_server():
    try:
        async with websockets.serve(handle_client, HOST, PORT, ping_timeout=30):
            print(f"Server listening on ws://{HOST}:{PORT}")
            await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        print("Server task cancelled. Shutting down.")
    finally:
        print("Server shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down.")
