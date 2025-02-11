# app/graph_handler.py
import pickle
import networkx as nx
from pathlib import Path

DATA_PATH = 'graph.pkl'

def load_graph():
    try:
        with open(DATA_PATH, 'rb') as file:
            G = pickle.load(file)
        print("Graph loaded successfully.")
    except FileNotFoundError:
        print("Pickle file not found. Initializing a new graph.")
        G = nx.Graph()
    except Exception as e:
        print(f"An error occurred while loading the graph: {e}")
        G = nx.Graph()
    return G

def save_graph(G):
    try:
        with open(DATA_PATH, 'wb') as file:
            pickle.dump(G, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Graph saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the graph: {e}")
