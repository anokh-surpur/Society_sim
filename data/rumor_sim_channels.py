import os
import random
import csv
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from google import genai


api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY environment variable")
client = genai.Client(api_key=api_key)


class Agent:
    def __init__(self, id, role, bias, trust, stubbornness=0.2):
        self.id = id
        self.role = role
        self.bias = bias
        self.trust = trust
        self.stubbornness = stubbornness
        self.belief = "neutral"
        self.trust_network = {}
        self.influence_score = 0

    def decide(self, rumor_text, sender=None, channel="direct", intervention=None):
        if intervention and random.random() < self.trust * intervention["strength"]:
            old = self.belief
            self.belief = "reject"
            if old != self.belief:
                return old, self.belief, f"fact-check via {intervention['source']}", "broadcast"
            return None

        if sender is not None:
            self.trust_network[sender] = self.trust_network.get(sender, 0) + 0.05
            self.trust = min(1.0, self.trust + self.trust_network[sender]*0.1)

        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"Agent role={self.role}, bias={self.bias}, rumor={rumor_text}"
            )
            decision = "believes" if "true" in response.text.lower() else "reject"
        except Exception:
            decision = "believes" if random.random() < (0.5 + 0.3*self.bias) else "reject"

        if random.random() < self.stubbornness:
            return None

        old = self.belief
        self.belief = decision
        if old != self.belief:
            return old, self.belief, "rumor evaluation", channel
        return None

# ----------------------------
# Communication Channels
# ----------------------------
def get_channel(agent):
    if agent.role == "family":
        return "whatsapp"
    elif agent.role == "media":
        return "broadcast"
    elif agent.role == "leader":
        return "public_speech"
    else:
        return random.choice(["direct", "whatsapp"])

# ----------------------------
# Simulation
# ----------------------------
def run_simulation(rumor, steps=5):
    agents = [
        Agent(0, "leader", bias=0.3, trust=0.9, stubbornness=0.1),
        Agent(1, "family", bias=-0.2, trust=0.7),
        Agent(2, "family", bias=0.5, trust=0.4),
        Agent(3, "media", bias=-0.3, trust=0.8),
        Agent(4, "friend", bias=0.1, trust=0.5),
        Agent(5, "friend", bias=0.0, trust=0.6),
        Agent(6, "leader", bias=0.2, trust=0.8),
        Agent(7, "family", bias=-0.1, trust=0.6),
        Agent(8, "media", bias=0.0, trust=0.7),
        Agent(9, "friend", bias=0.2, trust=0.5)
    ]

    fact_checks = [
        {"source": "Trusted Government News", "strength": 0.9}
    ]

    log = []
    history = []

    G = nx.Graph()
    for agent in agents:
        G.add_node(agent.id, role=agent.role)

    edges = [(0, 1), (0, 2), (0, 4), (0, 5), (1, 2), (3, 1), (3, 2), (3, 4), (3, 5), (4, 5), (6, 7), (6, 8), (7, 9)]
    G.add_edges_from(edges)

    with open("rumor_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "agent", "old_belief", "new_belief", "reason", "channel"])

        for step in range(steps):
            intervention = fact_checks[0] if step == 2 else None

            for agent in agents:
                channel = get_channel(agent)
                for sender in agents:
                    if sender.id != agent.id:
                        result = agent.decide(rumor, sender.id, channel, intervention)
                        if result:
                            old, new, reason, ch = result
                            writer.writerow([step, agent.id, old, new, reason, ch])
                            log.append((step, agent.id, old, new, reason, ch))
                            sender.influence_score += 1

            # Dynamic network update
            for (u, v) in list(G.edges):
                if agents[u].belief != agents[v].belief:
                    if random.random() < 0.3:
                        G.remove_edge(u, v)
                else:
                    if random.random() < 0.2:
                        G.add_edge(u, v)

            history.append({agent.id: agent.belief for agent in agents})

    return agents, history, log, G

# ----------------------------
# Animation
# ----------------------------
def animate_network(history, G):
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(step):
        ax.clear()
        colors = []
        for node in G.nodes():
            belief = history[step][node]
            if belief == "believes":
                colors.append("green")
            elif belief == "reject":
                colors.append("red")
            else:
                colors.append("gray")

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax, node_size=800)
        ax.set_title(f"Rumor Spread Step {step}")

    ani = FuncAnimation(fig, update, frames=len(history), interval=1000, repeat=False)
    ani.save("rumor_animation.gif", writer="pillow")
    plt.show()

# ----------------------------
# Run Simulation
# ----------------------------
if __name__ == "__main__":
    rumor = "The government is giving free land to all citizens"

    agents, history, log, G = run_simulation(rumor, steps=5)

    believers = sum(1 for a in agents if a.belief == "believes")
    rejecters = sum(1 for a in agents if a.belief == "reject")
    neutrals = sum(1 for a in agents if a.belief == "neutral")

    print("\n--- Simulation Summary ---")
    print(f"believers: {believers}")
    print(f"rejecters: {rejecters}")
    print(f"neutral: {neutrals}")
    print("Log saved to rumor_log.csv")

    for i, snap in enumerate(history):
        print(f"Step {i}: {snap}")

    animate_network(history, G)
