import numpy as np


class BaseAgent:
    def __init__(self, env, trajs_paths):
        self.env = env
        self.trajs_paths = trajs_paths
        self.test_mode = False

    def select_action(self, state):
        raise NotImplementedError

    def perform_action(self, action):
        next_state, reward, done, _ = self.env.step(action)

        return reward, next_state, done

    def rollout(self):
        state = self.env.reset()
        traj = []
        retvrn = 0

        done = False

        while not done:
            action = self.select_action(state)[0]
            reward, next_state, done = self.perform_action(action)

            traj += [(state, action)]
            retvrn += reward

            state = next_state

        return traj, retvrn

    def rollout_env(self, env_wrapper, eval_il):
        state = self.env.reset()
        state_env = env_wrapper._get_obs(state)

        orig_traj = []
        traj = []
        retvrn = 0

        done = False

        while not done:

            if eval_il:
                action = self.select_action(state_env)[0]
            else:
                action = self.select_action(state)[0]

            next_state, reward, done, _ = self.env.step(action)
            next_state_env = env_wrapper._get_obs(next_state)

            orig_traj += [(state, action)]
            traj += [(state_env, action, env_wrapper._idx)]
            retvrn += reward

            state = next_state
            state_env = next_state_env

        return traj, retvrn

    def test(self, num_episodes):
        self.test_mode = True

        trajs = []
        returns = []

        for episode_index in range(num_episodes):
            if episode_index % 100 == 0:
                print("episode: %d" % episode_index)

            traj, retvrn = self.rollout()

            trajs += [traj]
            returns += [retvrn]

        np.save(self.trajs_paths, {"trajs": trajs, "returns": returns})

        return np.mean(returns), np.std(returns)

    # NOTE: Unused.
    # def test_env(self, num_episodes, env_id, noise, mult_factor, eval_il):
    #     trajs = []
    #     returns = []

    #     env_wrapper = EnvWrapper(env=self.env, noise=noise, mult_factor=mult_factor,
    #                              idx=env_id, seed=1)

    #     for episode_index in range(num_episodes):
    #         if episode_index % 100 == 0:
    #             print('episode: %d' % episode_index)

    #         traj, retvrn = self.rollout_env(env_wrapper, eval_il)

    #         trajs += [traj]
    #         returns += [retvrn]

    #     print(self.trajs_paths[env_id])

    #     np.save(self.trajs_paths[env_id], {'trajs': trajs, 'returns': returns})

    #     return np.mean(returns), np.std(returns)
