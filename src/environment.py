import networkx as nx
import random
from src.agent import Agent

def build_environment(api_key):
    G = nx.erdos_renyi_graph(n=10, p=0.3, seed=42)  # 10 agents
    roles = ["Farmer", "Student", "Shopkeeper", "Teacher", "Policeman"]
    agents = {}

    for node in G.nodes():
        role = random.choice(roles)
        skepticism = random.uniform(0.2, 0.8)
        trust_weight = random.uniform(0.3, 1.0)
        agents[node] = Agent(
            name=f"Agent{node}",
            role=role,
            skepticism=skepticism,
            trust_weight=trust_weight,
            api_key=api_key
        )

    return G, agents
