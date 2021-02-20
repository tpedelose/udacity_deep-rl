[image1]: ./multiagent.png "Multiagent Training"

[image2]: ./rewards.png

# Udacity Deep Reinforcement Nanodegree
## Project 2 :: Continuous Control

## Algorithm and Methods
My initial attempt was to implement A2C in order to parallelize the learning process on the GPU.  I could not get the solo agent to learn -- my suspision is that my implementation was wrong.

I then moved on to repurposing the DDPG agent described in the code. Pulling everyhting over and adapting to the Reacher environment was simple enough.  The agent learned well enough in the solo environment that I decided to fiddle with the underlying network structure.  

I was curious to see if I could make the networks smaller with the hope that this would speed up the training step.  I changed the Actor's hidden layer to 64 parameters (down from 256).  I also changed the Critic by removing the fourth layer and reducing the two remaining hidden layers to 128 (from 256) and 64 (also from 256).  This did not seem to greatly affect performance and slightly increased training time to "solve" the environment.

Next, I moved on to implementing the multi-agent environment.  This was easily implemented due to the baseline code I was working with.  The agent needed to be updated to handle collection of more than one agent's state-action-reward-nextstate tuple at a time.  Since there is no need to worry about correlation with the replay buffer, iterating over this list of agents to add to the buffer is a simple solution.
I also modified the OUNoise method. Adding a size parameter to help the sampling process create noise for each of the agents.


### Env: Version 2
I decided to train my final DDPG network in the 20 agent environment. 

![image1]

I used the following hyperparameters in my solution:
```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
```

DDPG does not teach each of these agents individually.  In reality it gathers each of the agents' data into a replay buffer and learns after a threshold of data has been reached.  This means that a single policy is used for each of the agents.

Below you see a chart of the average reward of the 20 agents per episode during training of my submitted solution.

![image2]

The agent solved the environment (average score >= 30) at episode 203 with an average score of 30.06.  Some checkpoint values are provided below.
* Episode 100
    * Average Score: 8.84
    * Episode Score: 19.46
* Episode 200
    * Average Score: 29.59
    * Episode Score: 32.41
* Episode 202
    * Average Score: 29.91	
    * Episode Score: 36.54


## Future Work
I would like to go back and finish implementing the A3C/A2C algorithm.  This seems like an interesting algorithm that could be useful for online learning agents.  Specifically in situations where agents can not regularly communicate or have unreliable connections.