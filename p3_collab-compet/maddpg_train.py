from collections import deque
from os import SCHED_BATCH
import torch
import numpy as np
from pathlib import Path
from maddpg_agent import Agent


SOLVED_SCORE = 0.5
SOLVED_WINDOW = 100


def save_checkpoint(agent, dir):
    torch.save(agent.actor_local.state_dict(), dir / 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), dir / 'checkpoint_critic.pth')


def train(env, n_episodes=1000, seed=101, stop_on_solve=True):
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

    agents = Agent(state_size, action_size,
                   n_agents=n_agents, random_seed=seed)

    episode_scores = []
    window_means = []
    solved = np.NINF

    # Training
    for i_episode in range(1, n_episodes+1):
        score = np.zeros(n_agents)

        agents.reset()
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        while True:
            actions = agents.act(states)
            env_info = env.step(actions)[brain_name]

            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agents.step(states, actions, rewards, next_states, dones)

            states = next_states
            score += rewards

            if np.any(dones):
                break

        # Logging
        best_score = np.max(score)
        episode_scores.append(best_score)
        window_mean = np.mean(episode_scores[-SOLVED_WINDOW:])
        window_means.append(window_mean)

        episode_info = f'Episode {i_episode} \tAverage Score: {window_mean: .3f}'

        if len(episode_scores) >= SOLVED_WINDOW and window_mean >= SOLVED_SCORE:
            if window_mean > solved:
                if solved == np.NINF:
                    print("\rEnvironment solved! " + episode_info)
                else:
                    print("\rBeat previous solution! " + episode_info)

                save_checkpoint(agents, Path("./checkpoints"))
                solved = window_mean

            if stop_on_solve:
                break

        else:
            end = ""
            if i_episode % 100 == 0:
                end = "\n"
                if solved > np.NINF:
                    save_checkpoint(agents, Path("./checkpoints"))

            print("\r" + episode_info +
                  f" \tEpisode Score: {best_score:.2f}", end=end)

    return episode_scores, window_means


def evaluate(env, actor_checkpoint, n_episodes=1):
    # Setup
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]

    n_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    agent = Agent(state_size, action_size, n_agents=n_agents)
    agent.actor_local.load_state_dict(torch.load(actor_checkpoint))
    agent.eval()

    for i_episode in range(1, n_episodes+1):
        score = np.zeros(n_agents)

        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations

        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]

            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            states = next_states
            score += rewards

            if np.any(dones):
                break
