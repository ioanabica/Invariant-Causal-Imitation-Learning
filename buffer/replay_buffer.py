from .base_buffer import *


class ReplayBuffer(BaseBuffer):
    def sample(self):
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)

        return self._take_from(indices)
