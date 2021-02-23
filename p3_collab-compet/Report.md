[image1]: ./scores.png "Multiagent DDPG Tennis scores"

# Udacity Deep Reinforcement Nanodegree
## Project 2 :: Collaberation / Competition

## Algorithm and Methods

### Methods
I continued my attempts from project 2 to implement A3C/A2C, as I thought the inherit nature of the parallelized networks would make it easy to multi-agent use cases.  However, I was unable to get this algorithm running as the project due-date approached, so I pivoted to modifying my solution to project 2 using the DDPG algorithm.

### The MADDPG Algorithm
The base DDPG algorithm is an actor-critic method that uses target networks to reduce variance and stabilize training.  The actor approximates the optimal policy. The critic approximates the action-value function (expected future rewards for an action).  In training, the corresponding regular networks are updated at regular intervals based on their loss.  Using the `soft update` method, the target networks are gradually updated over these timesteps, in order to further reduce variance and converge faster.

Since the DDPG algorithm works with an experience replay buffer, which de-correlates sequential states, it can easily be turned into a multi-agent solution through self-play. Instead of acting once per timestep in an environment, act for the number of players in a game.  The resulting s,a,r,s' tuple will be stored in the buffer to be accessed at training time.  This would also make it easily extensible to parallelized environments.

### Model Structure and Hyperparameters
My initial attempt, the size 64 hidden layers for both the actor and critic from project 2 would not learn. After reverting to size 128 and getting the agent to solve the environment I continued trying to get smaller networks to work. I eventually stopped at 128, 92 for hidden layer sizes, that solved the environment in around 1000 episodes.

I used the following hyperparameters when training:
```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
```

## Results
My final agent solved the environment at episode 960, with an average score of 0.503.
After solve, I continued to let the agent run until episode 1500.  The best agent was saved at episode 1225 with average score of 0.608.  You can try this out by loading the `checkpoint_actor.pth` file with the `ddpg_train.evaluate()` function.

![image1]
Above: MADDPG Scores
* Blue - Episode best score
* Red - Moving average, past 100 episodes
* Pink - Episode solve threshold (0.5)


## Future Work
I really want to correctly implement the A3C/A2C algorithm to work on these and the other projects, to get a full understanding on how that algorithm works.  From looking at subsequent research, it may not be the best of the algorithms for RL, but I think that correctly implementing the parallelized algorithm will give me more insight on the underlying function of RL.