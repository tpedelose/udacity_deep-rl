import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D
import numpy as np
from collections import deque
import logging
from pathlib import Path
import time


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    NAME = "A3C"

    def __init__(self, state_size, action_size, lr_actor=1e-3, lr_critic=1e-3) -> None:
        self.state_size = state_size
        self.action_size = action_size

        self.actor = Actor(self.state_size, self.action_size, lr=lr_actor)
        self.critic = Critic(self.state_size, lr=lr_critic)

        self.memory = Memory()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)

        value = self.critic.forward(state)
        probs = self.actor.forward(state)
        dist = D.Categorical(probs)
        action = dist.sample()

        return action, value

    def learn(self, gamma):
        advantages = self.get_advantages(gamma)
        log_probabilities = self.get_log_probabilities()

        self.critic.backward(advantages)
        self.actor.backward(advantages.detach(), log_probabilities)

    def save_checkpoints(self, dir, brain_name=None):
        now = int(time.time())
        prefix = "_".join(filter(None.__ne__, [Agent.name, brain_name]))

        torch.save(
            self.actor.state_dict(),
            dir/f"{prefix}_actor_{now}.pth"
        )
        torch.save(
            self.critic.state_dict(),
            dir/f"{prefix}_critic_{now}.pth"
        )

    def load_checkpoints(self, actor_weights, critic_weights):
        raise NotImplementedError

    def get_advantages(self, gamma, n_steps=1):
        states = self.memory.states
        rewards = self.memory.rewards
        next_states = self.memory.next_states
        dones = self.memory.dones
        values = self.memory.values

        advantages = []
        returns = values[-1].detach()

        for i in reversed(range(n_steps)):
            advantage = rewards[i]
            + ((1.0 - dones[i]) * gamma * returns)
            - self.critic(
                torch.from_numpy(next_states[i]).float())
            advantages.append(advantage)

        # predicted_scores = self.critic(torch.from_numpy(states).float())
        # predicted_future_scores = self.critic(
        #     torch.from_numpy(next_states).float())

        # advantages = rewards
        # - predicted_scores
        # + ((1.0 - dones)
        #     * gamma
        #    * predicted_future_scores)

        return np.array(advantages)

    def get_log_probabilities(self):
        dists = self.memory.dists
        actions = dists.sample()
        log_probabilities = m.log_prob(actions)

        return log_probabilities


class Actor(nn.Module):
    def __init__(self, n_inputs, n_actions, lr=1e-3) -> None:
        super().__init__()

        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        _ = self.fc1(state)
        _ = F.relu(_)
        _ = self.fc2(_)
        _ = F.softmax(_)
        return _  # Action probability distribution

    def backward(self, advantages, log_probabilities):
        log_probs = torch.stack(self.memory.log_probabilities)
        loss = (- log_probabilities * advantages).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Critic(nn.Module):
    def __init__(self, n_inputs, lr=1e-3) -> None:
        super().__init__()

        # Critic looks at the current state and predicts the episode score (a single value)
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        _ = self.fc1(state)
        _ = F.relu(_)
        _ = self.fc2(_)
        return _  # Predicted score

    def backward(self, advantages):
        loss = 0.5 * advantages.detach().pow(2).mean()  # MSE
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.optimizer.step()


class Memory():
    def __init__(self) -> None:
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None

        # self.log_probabilities = None
        # self.advantages = None

    def add(self, s, a, r, ns):
        self.states.append(s, axis=0)
        self.actions.append(a, axis=0)
        self.rewards.append(r, axis=0)
        self.next_states.append(ns, axis=0)

    def clear(self):
        self.__init__()


class DataLogger():
    def __init__(self) -> None:
        scores = []

    def add(tuples):
        for t in tuples:
            pass

    def window():
        pass


def train(env, n_episodes=100, max_timesteps=1000, n_bootstrap=1, window_size=100, gamma=0.99, seed=101):
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]

    n_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    # Todo:  Generalize to N agents
    # agent = Agent(state_size, action_size)
    agents = [Agent(state_size, action_size) for _ in range(n_agents)]
    scores = np.zeros(n_agents)

    # Training
    for i_episode in range(1, n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        for timestep in range(max_timesteps):
            # action_probs = np.concatenate([
            #     agent.act(state).detach().cpu().numpy()
            #     for agent, state in zip(agents, states)
            # ])

            actions = []
            action, value = zip(*[
                agent.act(state)
                for agent, state in zip(agents, states)
            ])

            actions = np.reshape(actions, (n_agents, action_size))
            env_info = env.step(actions)[brain_name]

            next_states = env_info.vector_observations
            rewards = env_info.rewards

            agent.steps()
            scores.append(rewards, axis=0)

            if len(agent.memory) >= n_bootstrap or timestep == max_timesteps or np.any(dones):
                agent.learn(gamma)
                agent.storage.clear()

            if np.any(dones):
                break

        # Logging
        episode_mean = np.mean(scores)
        window_mean = np.mean(scores[window_size:])

        episode_info = f'\rEpisode {i_episode} \tAverage Score: {window_mean: .2f}'

        if scores.shape[0] >= window_size and window_mean >= 30.0:
            print("Environment solved! " + episode_info)
            agent.save_checkpoints(Path("./checkpoints"))
            break

        elif i_episode % 100 == 0:
            print(episode_info + f" \tEpisode Score: {episode_mean:.2f}")
            agent.save_checkpoints(Path("./checkpoints"))

        else:
            print(episode_info +
                  f" \tEpisode Score: {episode_mean:.2f}", end="")

    return scores, agent
