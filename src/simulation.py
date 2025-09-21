import random
import pandas as pd
import matplotlib.pyplot as plt
from src.environment import build_environment

class RumorSimulation:
    def __init__(self, api_key, rumor, truth_label):
        self.api_key = api_key
        self.rumor = rumor
        self.truth_label = truth_label   # True or False
        self.logs = []

    def run(self, steps=5):
        G, agents = build_environment(self.api_key)
        seed = random.choice(list(agents.keys()))
        active = [seed]

        for t in range(steps):
            new_active = []
            for node in active:
                agent = agents[node]
                belief = agent.think(self.rumor, source="seed", truth_label=self.truth_label)
                self.logs.append({"time": t, "agent": agent.name, "belief": belief})

                if belief == 1:  # only spread if believed
                    neighbors = list(G.neighbors(node))
                    new_active.extend(neighbors)

            active = list(set(new_active))

        df = pd.DataFrame(self.logs)
        df.to_csv("data/results.csv", index=False)
        self.visualize(df)
        return df

    def visualize(self, df):
        plt.figure(figsize=(6,4))
        believers = df.groupby("time")["belief"].mean()
        plt.plot(believers.index, believers.values, marker="o")
        plt.xlabel("Time")
        plt.ylabel("% Believers")
        plt.title("Rumor Spread Over Time")
        plt.savefig("results/plots.png")
        plt.show()
