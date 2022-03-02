from .base_agent import BaseAgent


class TrainableAgent(BaseAgent):
    def train(self):
        self.test_mode = False
