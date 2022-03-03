from .trainable_agent import TrainableAgent


class SerializableAgent(TrainableAgent):
    def __init__(
        self,
        env,
        trajs_paths,
        model_path,
    ):
        super(SerializableAgent, self).__init__(
            env=env,
            trajs_paths=trajs_paths,
        )

        self.model_path = model_path

    def serialize(self):
        raise NotImplementedError

    def deserialize(self):
        raise NotImplementedError
