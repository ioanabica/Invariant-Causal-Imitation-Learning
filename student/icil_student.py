import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from agent import CUDAAgent

from .base_student import BaseStudent


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def flatten(_list):
    return [item for sublist in _list for item in sublist]


# pylint: disable=arguments-differ
class ICILStudent(BaseStudent, CUDAAgent):
    def __init__(
        self,
        env,
        trajs_paths,
        model_path,
        num_training_envs,
        teacher,
        causal_features_encoder,
        noise_features_encoders,
        causal_features_decoder,
        noise_features_decoders,
        observations_decoder,
        env_discriminator,
        policy_network,
        energy_model,
        mine_network,
        buffer,
        adam_alpha,
    ):
        super(ICILStudent, self).__init__(
            env=env,
            trajs_paths=trajs_paths,
            model_path=model_path,
            teacher=teacher,
            buffer=buffer,
        )

        self.num_training_envs = num_training_envs

        self.causal_features_encoder = causal_features_encoder.to(self.device)
        self.causal_features_decoder = causal_features_decoder.to(self.device)
        self.observations_decoder = observations_decoder.to(self.device)
        self.env_discriminator = env_discriminator.to(self.device)
        self.policy_network = policy_network.to(self.device)
        self.mine_network = mine_network.to(self.device)

        self.energy_model = energy_model

        self.noise_features_encoders = []
        self.noise_features_decoders = []
        for i in range(self.num_training_envs):
            self.noise_features_encoders.append(noise_features_encoders[i].to(self.device))
            self.noise_features_decoders.append(noise_features_decoders[i].to(self.device))

        self.adam_alpha = adam_alpha

        noise_models_params = flatten(
            [list(noise_features_encoder.parameters()) for noise_features_encoder in self.noise_features_encoders]
        ) + flatten(
            [list(noise_features_decoder.parameters()) for noise_features_decoder in self.noise_features_decoders]
        )

        self.rep_optimizer = optim.Adam(
            list(causal_features_encoder.parameters())
            + list(causal_features_decoder.parameters())
            + list(observations_decoder.parameters())
            + noise_models_params
            + list(policy_network.parameters()),
            lr=self.adam_alpha,
        )

        self.policy_opt = optim.Adam(
            list(causal_features_encoder.parameters()) + list(policy_network.parameters()), lr=self.adam_alpha
        )

        self.disc_opt = optim.Adam(list(env_discriminator.parameters()), lr=self.adam_alpha)

        self.mine_opt = optim.Adam(self.mine_network.parameters(), lr=1e-4)

        self.buffer = buffer

    def select_action(self, state, eval_mode=False):
        causal_rep = self.causal_features_encoder(torch.FloatTensor(state).to(self.device))
        action = self.policy_network(causal_rep).argmax()
        action = action.detach().cpu().numpy()

        if eval_mode:
            action = self.policy_network(causal_rep).detach().cpu().numpy()
            num_actions = action.shape[0]
            action = np.argmax(action)
            one_hot_action = np.eye(num_actions)[action]

            action_logits = self.policy_network(causal_rep).detach().cpu().numpy()
            action_prob = softmax(action_logits)
            return one_hot_action, action_prob

        return action

    def train(self, num_updates):

        for update_index in tqdm(range(num_updates)):
            self._update_networks()

        self.env.close()

        self.serialize()

    def serialize(self):
        torch.save(self.policy_network.state_dict(), self.model_path)

    def deserialize(self):
        self.policy_network.load_state_dict(torch.load(self.model_path))

    def _update_networks(self):
        samples = self.buffer.sample()

        (
            ce_loss,
            disc_entropy,
            next_state_pred_loss,
            next_state_energy_loss,
            expert_samples_energy,
            mi_loss,
            env_discriminator_loss,
            mine_loss,
        ) = self._compute_loss(samples)

        rep_loss = disc_entropy + next_state_pred_loss + mi_loss
        policy_loss = ce_loss + next_state_energy_loss

        self.rep_optimizer.zero_grad()
        self.policy_opt.zero_grad()
        rep_loss.backward(retain_graph=True)
        policy_loss.backward()
        self.rep_optimizer.step()
        self.policy_opt.step()

        self.disc_opt.zero_grad()
        env_discriminator_loss.backward()
        self.disc_opt.step()

        self.mine_opt.zero_grad()
        mine_loss.backward()
        self.mine_opt.step()

    def _compute_loss(self, samples):
        state = torch.FloatTensor(samples["state"]).to(self.device)
        action = torch.LongTensor(samples["action"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_state"]).to(self.device)
        env_ids = torch.LongTensor(samples["env"]).to(self.device)

        causal_rep = self.causal_features_encoder(state)

        # 1. Policy loss
        qvalues = self.policy_network(causal_rep)
        ce_loss = nn.CrossEntropyLoss()(qvalues, action)

        action_one_hot = F.one_hot(action).type(torch.FloatTensor).to(self.device)
        imitation_action = F.gumbel_softmax(qvalues, hard=True).type(torch.FloatTensor).to(self.device)

        # 2. Env discriminator entropy loss for causal representation learning
        predicted_env = self.env_discriminator(causal_rep)
        disc_entropy_entropy = torch.mean(F.softmax(predicted_env, dim=1) * F.log_softmax(predicted_env, dim=1))

        # 3. Enc discriminator cross-entropy loss for training environment classifier
        predicted_env = self.env_discriminator(causal_rep.detach())
        env_discriminator_loss = nn.CrossEntropyLoss()(predicted_env, env_ids)

        # 4. Next state prediction loss
        #############################################################################################################
        noise_rep = causal_rep.clone()
        next_state_noise_rep = causal_rep.clone()

        for env_id in range(self.num_training_envs):
            env_samples_idx = torch.where(env_ids == env_id)[0]

            if env_samples_idx.shape[0] == 0:
                env_samples_idx = torch.LongTensor([0]).to(self.device)

            state_env = state[env_samples_idx]
            action_one_hot_env = action_one_hot[env_samples_idx]

            noise_rep_env = self.noise_features_encoders[env_id](state_env)
            next_state_noise_rep_env = self.noise_features_decoders[env_id](noise_rep_env, action_one_hot_env)

            noise_rep[env_samples_idx] = noise_rep_env
            next_state_noise_rep[env_samples_idx] = next_state_noise_rep_env

        next_state_causal_rep = self.causal_features_decoder(causal_rep, action_one_hot)
        predicted_next_state = self.observations_decoder(next_state_causal_rep, next_state_noise_rep)
        next_state_pred_loss = nn.MSELoss()(predicted_next_state, next_state)
        #############################################################################################################

        # 5. Mutual information loss
        mi_loss = self.mine_network.mi(causal_rep, noise_rep)
        mine_loss = self.mine_network.forward(causal_rep.detach(), noise_rep.detach())

        # 6. Energy loss
        #############################################################################################################
        next_state_noise_rep_energy = causal_rep.clone()
        with torch.no_grad():
            for env_id in range(self.num_training_envs):
                env_samples_idx = torch.where(env_ids == env_id)[0]

                if env_samples_idx.shape[0] == 0:
                    env_samples_idx = torch.LongTensor([0]).to(self.device)

                imitation_action_env = imitation_action[env_samples_idx]
                noise_rep_env = noise_rep[env_samples_idx]

                next_state_noise_rep_env = self.noise_features_decoders[env_id](noise_rep_env, imitation_action_env)
                next_state_noise_rep_energy[env_samples_idx] = next_state_noise_rep_env

            next_state_causal_rep_energy = self.causal_features_decoder(causal_rep, imitation_action)
            predicted_next_state_energy = self.observations_decoder(
                next_state_causal_rep_energy, next_state_noise_rep_energy
            )

        expert_samples_energy = self.energy_model.forward(state).mean()
        next_state_energy_loss = self.energy_model.forward(predicted_next_state_energy).mean()
        #############################################################################################################

        return (
            ce_loss,
            disc_entropy_entropy,
            next_state_pred_loss,
            next_state_energy_loss,
            expert_samples_energy,
            mi_loss,
            env_discriminator_loss,
            mine_loss,
        )
