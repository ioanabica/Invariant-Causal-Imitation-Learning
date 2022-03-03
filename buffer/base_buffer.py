import numpy as np


class BaseBuffer:
    def __init__(self, state_dim, total_size, batch_size):
        self.state_buf = np.zeros([total_size, state_dim], dtype=np.float32)
        self.action_buf = np.zeros([total_size], dtype=np.float32)
        self.reward_buf = np.zeros([total_size], dtype=np.float32)
        self.next_state_buf = np.zeros([total_size, state_dim], dtype=np.float32)
        self.done_buf = np.zeros(total_size, dtype=np.float32)
        self.env_buf = np.zeros(total_size, dtype=np.float32)

        self.total_size = total_size
        self.batch_size = batch_size

        self.ptr = 0
        self.size = 0

    def store(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        env,
    ):
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_state_buf[self.ptr] = next_state
        self.done_buf[self.ptr] = done
        self.env_buf[self.ptr] = env

        self.ptr = (self.ptr + 1) % self.total_size
        self.size = min(self.size + 1, self.total_size)

    def sample(self):
        raise NotImplementedError

    def sample_all(self):
        return self._take_from(slice(None, None, None))

    def _take_from(self, indices):
        return dict(
            state=self.state_buf[indices],
            action=self.action_buf[indices],
            reward=self.reward_buf[indices],
            next_state=self.next_state_buf[indices],
            done=self.done_buf[indices],
            env=self.env_buf[indices],
        )

    def __len__(self):
        return self.size
