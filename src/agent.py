import random
import google.generativeai as genai

class Agent:
    def __init__(self, name, role, skepticism, trust_weight, api_key):
        self.name = name
        self.role = role
        self.skepticism = skepticism   # 0-1 (higher = more likely to fact-check)
        self.trust_weight = trust_weight
        self.memory = []               # [(rumor, belief, source)]
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def think(self, rumor, source, truth_label=None):
        """
        Ask Gemini: does the agent believe this rumor?
        """
        prompt = f"""
        You are {self.role} named {self.name}.
        Rumor received: "{rumor}"
        Skepticism: {self.skepticism}
        Source: {source}

        Decide: Do you believe it? Answer 'BELIEVE' or 'DOUBT'.
        """
        try:
            response = self.model.generate_content(prompt).text.strip()
        except Exception as e:
            response = "DOUBT"

        belief = 1 if "BELIEVE" in response.upper() else 0
        self.memory.append((rumor, belief, source))
        return belief
