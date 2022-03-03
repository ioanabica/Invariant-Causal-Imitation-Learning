from .base_agent import BaseAgent


class TrainableAgent(BaseAgent):  # pylint: disable=abstract-method
    def train(self):
        self.test_mode = False
