import socket
import threading
import pickle
import sys
import torch.nn as nn
import sqlite3
import torch
from fl_strategy import FL_Strategy
from fl_plan import FL_Plan
from client_model import ClientModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
import time


# Events to ensure synchronization
weights_recvd_event = threading.Event()

CURRENT_EPOCH = 0
# Mixing parameter for personalization
LAMBDA = 0.9

class Server:
    def __init__(self, server_ip, server_port):
        self.connected_clients = {}
        self.trained_clients = []
        self.pretrained_clients = []
        self.cluster_dict = {}
        self.ip = server_ip
        self.port = int(server_port)
        self.server_db_path = 'server_data/server_db.db'
        self.event_dict = {'UPDATED_WEIGHTS': weights_recvd_event}
        self.device = self.get_device()
        print(f'Using {self.device}')
        torch.manual_seed(32)
        self.client_model = ClientModel()
        self.recvd_initial_weights = 0

    def create_db_schema(self):
        """
        Creates the server-side database schema.
        """
        clients_table = """
        CREATE TABLE clients(
            id INT PRIMARY KEY,
            ip VARCHAR(50),
            port INT,
            datasize INT,
            cluster_id INT
        )
        """
        training_table = """
        CREATE TABLE training(
            client_id INT,
            epoch INT,
            model_updated_weights BLOB,
            model_aggregated_weights BLOB,
            PRIMARY KEY (client_id, epoch),
            FOREIGN KEY (client_id) REFERENCES clients (id)
        )
        """
        epoch_stats_table = """
        CREATE TABLE epoch_stats(
            epoch INT PRIMARY KEY,
            connected_clients INT,
            trained_clients INT
        )
        """
        self.execute_query(query=clients_table) if not self.check_table_existence(target_table='clients') else None
        self.execute_query(query=training_table) if not self.check_table_existence(target_table='training') else None
        self.execute_query(query=epoch_stats_table) if not self.check_table_existence(target_table='epoch_stats') else None
        print('[+] Database schema created/loaded successsfully')

    def check_table_existence(self, target_table: str) -> bool:
        """
        Checks if a specific table exists within the database.
        Args:
            target_table: Table to look for.
        Returns:
            True or False depending on existense.
        """
        query = "SELECT name FROM sqlite_master WHERE type ='table'"
        tables = self.execute_query(query=query, values=None, fetch_data_flag=True, fetch_all_flag=True)
        exists = any(table[0] == target_table for table in tables) if tables is not None else False
        return exists
    
    def execute_query(self, query: str, values=None, fetch_data_flag=False, fetch_all_flag=False):
        """
        Executes a given query. Either for retrieval or update purposes.
        Args:
            query: Query to be executed
            values: Query values
            fetch_data_flag: Flag that signals a retrieval query
            fetch_all_flag: Flag that signals retrieval of all table data or just the first row.
        Returns:
            The data fetched for a specified query. If it is not a retrieval query then None is returned. 
        """
        try:
            connection = sqlite3.Connection(self.server_db_path)
            cursor = connection.cursor()
            cursor.execute(query, values) if values is not None else cursor.execute(query)
            fetched_data = (cursor.fetchall() if fetch_all_flag else cursor.fetchone()[0]) if fetch_data_flag else None
            connection.commit()
            connection.close()        
            return fetched_data
        except sqlite3.Error as error:
            print(f'{query} \nFailed with error:\n{error}')

    def create_socket(self):
        """
        Binds the server-side socket to enable communication.
        """
        try:
            self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.server_socket.bind((self.ip, self.port))
            print(f'[+] Server initialized successfully at {self.ip, self.port}')
        except socket.error as error:
            print(f'Socket initialization failed with error:\n{error}')
            sys.exit(0)

    def listen_for_connections(self):
        """
        Listening to the server-side port for incoming connections from clients.
        Creates a unique communication thread for each connected client. 
        """
        try:
            self.server_socket.listen()
            while True:
                client_socket, client_address = self.server_socket.accept()
                client_id = self.handle_connections(client_address, client_socket)
                threading.Thread(target=self.listen_for_messages, args=(client_socket, client_id)).start()
        except socket.error as error:
            print(f'Connection handling thread failed:\n{error}')
            
    def listen_for_messages(self, client_socket: socket.socket, client_id: int):
        """
        Client-specific communication thread. Listens for incoming messages from a unique client.
        Args:
            client_socket: socket used from a particular client to establish communication.
        """
        data_packet = b''
        try:
            while True:
                data_chunk = client_socket.recv(4096)
                data_packet += data_chunk
                if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                        threading.Thread(target=self.handle_data, args=(data_packet, client_id)).start()
                        data_packet = b'' 
                if not data_chunk:
                    break
        except socket.error as error:
            # Handle client dropout
            print(f'Error receiving data from {client_id, self.connected_clients[client_id][0]}:\n{error}')
            client_socket.close()
            self.connected_clients.pop(client_id)
            self.trained_clients.remove(client_id) if client_id in self.trained_clients else None

    
    def handle_connections(self, client_address: tuple, client_socket: socket.socket):
        """
        When a client connects -> Add him on db if nonexistent, append to connected_clients list and transmit initial weights
        Args: Tuple (client_ip, client_port)
        """
        client_ip, client_port = client_address
        query = """
        SELECT id
        FROM clients
        WHERE ip = ? AND port = ?
        """
        exists = self.execute_query(query=query, values=(client_ip, client_port), fetch_data_flag=True, fetch_all_flag=True)
        if len(exists) == 0:
            query = """
            SELECT id FROM clients ORDER BY id DESC LIMIT 1;
            """
            last_id = self.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
            client_id = 1 if len(last_id) == 0 else last_id[0][0] + 1
            query = """
            INSERT INTO clients (id, ip, port) VALUES (?, ?, ?)
            """
            self.execute_query(query=query, values=(client_id, client_ip, client_port))
        else:
            client_id = exists[0][0]
        self.connected_clients[client_id] = (client_address, client_socket)
        print(f'[+] Client {client_id, client_address} connected -> Connected clients: {len(self.connected_clients)}')
        #self.send_packet(data={'INITIAL_WEIGHTS': [self.client_model.state_dict(), self.classifier_model.state_dict()]}, client_socket=client_socket)
        self.send_packet(data={'PLAN': self.plan}, client_socket=client_socket)
        print(f'[+] Transmitted FL plan to client {client_id, client_address}')
        return client_id

    def send_packet(self, data: dict, client_socket: socket.socket):
        """
        Packs and sends a payload of data to a specified client.
        The format used is <START>DATA<END>, where DATA is a dictionary whose key is the header and whose value is the payload.
        Args:
            data: payload of data to be sent.
            client_socket: socket used for the communication with a specific client.
        """
        try:
            client_socket.sendall(b'<START>' + pickle.dumps(data) + b'<END>')
        except socket.error as error:
            print(f'Message sending failed with error:\n{error}')
            client_socket.close()

    def handle_data(self, data: dict, client_id: int):
        """
        Handles a received data packet according to its contents.
        A packet can be either be:
            2. Updated model weights during training
            3. Number of data samples before training
        Args:
            data: Dictionary where the key is the header and the value is the payload
            client_id: The id of the sending client
        """
        # Get payload and header
        data = pickle.loads(data.split(b'<START>')[1].split(b'<END>')[0])
        header = list(data.keys())[0]
        if header == 'UPDATED_WEIGHTS':
            query = """
            INSERT INTO training (client_id, epoch, model_updated_weights) VALUES (?, ?, ?)
            ON CONFLICT (client_id, epoch) DO
            UPDATE SET model_updated_weights = ?"""
            serialized_model_weights = pickle.dumps(data[header])
            self.execute_query(query=query, values=(client_id, CURRENT_EPOCH, serialized_model_weights, serialized_model_weights))
            self.trained_clients.append(client_id)
            print(f"\t[+] Received updated weights of client: {client_id, self.connected_clients[client_id][0]}")
            print(f"\t[+] Currently trained clients: {len(self.trained_clients)} / {self.strategy.MIN_PARTICIPANTS_FIT}")
        elif header == 'DATASIZE':
            query = "UPDATE clients SET datasize = ? WHERE id = ?"
            self.execute_query(query=query, values=(data[header], int(client_id)))
            self.pretrained_clients.append(client_id)
            print(f"\t[+] Received datasize of client: {client_id, self.connected_clients[client_id][0]}")
            print(f"\t[+] Currently awaiting clients: {len(self.pretrained_clients)} / {self.strategy.MIN_PARTICIPANTS_FIT}")
            
        self.event_dict[header].set() if header in self.event_dict.keys() else None

    def initialize_strategy(self, config_file_path: str):
        """
        Initializes the FL Strategy and FL Plan objects based on the configuration file.
        Args:
            config_file_path: The path to the configuration file
        """
        self.strategy = FL_Strategy(config_file=config_file_path)
        self.plan = FL_Plan(epochs=self.strategy.GLOBAL_TRAINING_ROUNDS, pre_epochs=self.strategy.PRETRAIN_ROUNDS, lr=self.strategy.LEARNING_RATE,
                            loss=self.strategy.CRITERION, optimizer=self.strategy.OPTIMIZER, batch_size=self.strategy.BATCH_SIZE,
                            model_weights=self.client_model.state_dict())
        print(f"[+] Emloyed Strategy:\n{self.strategy}")
        
    def get_device(self):
        """
        Check available devices (cuda or cpu).
        Returns: A torch.device() Object
        """
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def personalized_aggregation(self, needed_clients: list):
        """
        Aggregate personalized models for each client by mixing its locally trained model 
        with the global (FedAvg) model using a mixing parameter LAMBDA.
        """
        global_model = self.federated_averaging(needed_clients=needed_clients)
        #print(global_model.keys())
        for cid in needed_clients:
            query = "SELECT model_updated_weights FROM training WHERE client_id = ? AND epoch = ?"
            #print(cid, CURRENT_EPOCH)
            client_weights = pickle.loads(self.execute_query(query=query, values=(cid, CURRENT_EPOCH), fetch_data_flag=True))
            personalized_model = {}
            # For each layer's parameters
            for key in client_weights.keys():
                # Mix the global and local model based on LAMBDA
                personalized_model[key] = LAMBDA * client_weights[key] + (1 - LAMBDA) * global_model[key]     
                # Send personalized model to client
            self.send_packet(data={'AGGR_MODEL': personalized_model}, client_socket=self.connected_clients[cid][1])
            # Save on db
            query = "UPDATE training SET model_aggregated_weights = ? WHERE client_id = ? AND epoch = ?"
            self.execute_query(query=query, values=(pickle.dumps(personalized_model), cid, CURRENT_EPOCH))
            print([f'[+]Aggregated and transmitted personalized weights for client: {cid, self.connected_clients[cid][0]}'])

    def federated_averaging(self, needed_clients: list):
        """
        Implementation of the federated averaging algotithm.
        Args:
            needed_clients: A list with the IDs of this epoch's participating clients
        Returns:
            avg_weights: The global model obtained by Federated Averaging
        """
        # Fetch the updated model weights of all trained clients
        query = "SELECT model_updated_weights FROM training WHERE client_id IN (" + ", ".join(str(id) for id in needed_clients) + ") AND epoch = ?"
        all_client_model_weights = self.execute_query(query=query, values=(CURRENT_EPOCH, ), fetch_data_flag=True, fetch_all_flag=True)
        # Fetch the datasizes of all trained clients
        query = "SELECT datasize FROM clients WHERE id IN (" + ", ".join(str(id) for id in needed_clients) + ")"
        datasizes = self.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
        datasizes = [int(row[0]) for row in datasizes]
        # Dictionary for the global (averaged) weights
        avg_weights = {}
        # Calculate the data size of all the participating clients
        total_data = sum(datasize for datasize in datasizes)
        # For each client's updated weights
        for i in range(len(all_client_model_weights)):
            client_weight_dict = pickle.loads(all_client_model_weights[i][0])
            # For each layer's parameters
            for key in client_weight_dict.keys():
                # Average the weights and normalize based on client's contribution to total data size
                if key in avg_weights.keys():
                    avg_weights[key] += client_weight_dict[key] * (datasizes[i] / total_data)
                else:
                    avg_weights[key] = client_weight_dict[key] * (datasizes[i] / total_data)
        return avg_weights


if __name__ == '__main__':
# To execute, server_ip and server_port must be specified from the cl.
    if len(sys.argv) != 3:
        print('Incorrect number of command-line arguments.\nTo execute, server_ip and server_port must be specified from the cl.')
        sys.exit(1)

    server = Server(sys.argv[1], sys.argv[2])
    server.create_socket()
    server.create_db_schema()
    threading.Thread(target=server.listen_for_connections, args=()).start()
    server.initialize_strategy(config_file_path='strategy_config.txt')

    # Wait until a sufficient number of clients have connected
    while len(server.connected_clients) < server.strategy.MIN_PARTICIPANTS_START:
        pass
    # Start global training
    for e in range(server.strategy.GLOBAL_TRAINING_ROUNDS):
        CURRENT_EPOCH = e
        print(f'[+] Global training round {e + 1} initiated')
        # Track some statistics
        query = "INSERT INTO epoch_stats (epoch, connected_clients) VALUES (?, ?) ON CONFLICT (epoch) DO UPDATE SET connected_clients = ?"
        server.execute_query(query=query, values=(CURRENT_EPOCH, len(server.connected_clients), len(server.connected_clients)))
        
        # Wait to receive model updates from the minimum number of clients to aggregate
        while len(server.trained_clients) < server.strategy.MIN_PARTICIPANTS_FIT:
            pass
        #Aggregate personalized models
        server.personalized_aggregation(needed_clients=server.trained_clients)
        server.trained_clients.clear()