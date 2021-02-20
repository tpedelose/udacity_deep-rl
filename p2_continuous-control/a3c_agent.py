import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D
import numpy as np
from collections import deque


class A2C_Agent():
    def __init__(self, env, actor_hidden_size, critic_hidden_size, learning_rate=0.001) -> None:

        # Environment
        self.env = env
        self.brain_name = env.brain_names[0]
        env_info = env.reset()[self.brain_name]

        # self.n_selves = len(env_info.agents)
        # print('Number of agents:', self.n_selves)

        self.action_size = env.brains[self.brain_name].vector_action_space_size
        print('Size of each action:', self.action_size)

        states = env_info.vector_observations
        self.state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(
            states.shape[0], self.state_size))

        # Networks
        self.actor = A2C_Actor(
            self.state_size, self.action_size, actor_hidden_size)
        self.critic = A2C_Critic(
            self.state_size, critic_hidden_size)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate)

        # Storage
        self.memory = Memory()
        # self.logger = DataLogger()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        prediction_score = self.critic.forward(state)
        action_distribution = self.actor.forward(state)
        return prediction_score, action_distribution

    def _act(self, states):
        prediction_scores = self.critic.forward(states)
        action_distributions = self.actor.forward(states)

        m = D.Categorical(action_distributions)
        actions = m.sample()

        env_info = self.env.step(action_distributions.detach().data.numpy())[
            self.brain_name]                # send all actions to the environment

        rewards = env_info.rewards          # get reward (for each agent)
        dones = env_info.local_done         # see if episode finished
        # get next state (for each agent)
        next_states = env_info.vector_observations

        # Save data
        self.logger.log(scores=rewards)     # update the score (for each agent)
        self.memory.save(
            action_dists=action_distributions,
            prediction_score=prediction_scores,
            rewards=rewards,
            dones=dones)

        return pred_score, action_dist

    def learn(self):
        # state = torch.from_numpy(state).float()
        # next_state = torch.from_numpy(next_state).float()

        for i, (_, _, reward, done) in enumerate(memory.reversed()):

            values = torch.stack(memory.values)

        advantage = reward - \
            self.critic(state) + \
            ((1.0 - done) * gamma *
                self.critic())

        # Critic Learning
        critic_loss = advantage.pow(2).mean()  # MSE
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
        # actor_loss = - m.log_prob(action) * advantage.detach()
        actor_loss = (-torch.stack(self.memory.log_probs)
                      * advantage.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


class A2C_Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size) -> None:
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        _ = F.relu(self.fc1(state))
        action_dist = F.softmax(self.fc2(_))
        return action_dist


class A2C_Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size) -> None:
        super().__init__()

        # Critic looks at the current state and predicts the episode score (a single value)
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        _ = F.relu(self.fc1(state))
        score = self.fc2(_)
        return score


class Memory:
    def __init__(self) -> None:
        self.log_probs = []
        self.advantages = []

    def save(self, log_prob, advantage):
        self.log_probs.append(log_prob)
        self.advantages.append(advantage)

    def clear(self):
        self.log_probs = []
        self.advantages = []

    def __len__(self):
        return len(self.log_probs)


class DataLogger:
    def __init__(self) -> None:
        # self.scores_window =
        # self.episode_scores =
        pass

    def log(score):
        # self.scores_window
        pass


def train_A2C(env, brain_name, agent, n_episodes=500, max_timesteps=500, bootstrap=1, gamma=0.99):

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for episode in range(n_episodes):

        env_info = env.reset(train_mode=True)[
            brain_name]       # reset the environment
        # get the current state (for each agent)
        state = env_info.vector_observations[0]
        score = 0

        for timestep in range(max_timesteps):
            pred_score, action_dist = agent.act(state)
            m = D.Categorical(action_dist)
            action = m.sample()

            env_info = env.step(action_dist.detach().data.numpy())[
                brain_name]            # send all actions to the environment
            # get reward (for each agent)
            reward = env_info.rewards[0]
            # update the score (for each agent)
            score += reward
            # see if episode finished
            done = env_info.local_done[0]
            # get next state (for each agent)
            next_state = env_info.vector_observations[0]

            advantage = reward - \
                agent.critic(torch.from_numpy(state).float()) + \
                ((1.0 - done) * gamma *
                 agent.critic(torch.from_numpy(next_state).float()))

            agent.memory.save(
                log_prob=m.log_prob(action),
                advantage=advantage,
            )

            if len(agent.memory) >= bootstrap or timestep == max_timesteps or done:

                ### TRAIN ###
                advantages = torch.stack(
                    list(reversed(agent.memory.advantages)))
                log_probs = torch.stack(list(reversed(agent.memory.log_probs)))

                # Critic Learning
                critic_loss = 0.5 * advantages.pow(2).mean()  # MSE
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm(agent.critic.parameters(), 1)
                agent.critic_optimizer.step()

                # Actor Learning
                # https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
                actor_loss = - (log_probs * advantage.detach()).mean()
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                agent.actor_optimizer.step()

                # Wipe memory
                agent.memory.clear()

            if done:  # exit loop if episode finished
                break

            # roll over states to next time step
            state = next_state

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        # Console Logging
        if episode % 100 == 0:
            print(
                f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}')

        if np.mean(scores_window) >= 30.0:
            print(
                f'\nEnvironment solved in {episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            # torch.save(agent.qnetwork_local.state_dict(), f'checkpoint{int(time.time())}.pth')
            break

        else:
            print(
                f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")

    return scores
