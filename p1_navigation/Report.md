# Project Report

This projects implements a Double DQN system in order to train an agent to pick up yellow bananas whilst avoiding blue bananas.  The Double DQN works by implementing two seperate Q-networks -- one to keep track of the current action-values and another to describe the target of the prior network. 

## Hyperparameter Selection

For this implementation, I chose the following hyperparameters: 
* number of episodes = 500 
    - The number of episodes was a heuristic from seeing how previous runs performed.The final model solved the problem (average score of 13 over 100 episodes) at episode 432.

* max timesteps = 2000 
    - The max timesteps at 2000 was in an attempt to not cut-off well-performing agents too early.

* epsilon max = 1.0,  epsilon min = 0.01,  epsilon decay rate = 0.985
    - Epsilon starts at 1.0, allowing for full eploration of all actions to start, but quickly decays at rate 0.985 per episode to 0.01 at episode 306.  This allows the agent to exploit what it has learned while still exploring 10% of the time. The decay rate could probably be reduced further, to see if the agent will be able to solve the problem in fewer episodes.  You can examine the value of epsilon over time below:
    ![Value of epsilon over time](epsilon-decay.png?raw=true "Epsilon Decay")


## Rewards
Here you can see a plot of the rewards over the episode run:
![Score per episode](scores-vs-episodes.png?raw=true "Rewards over Episodes")


## Future Work
Since Double DQN was the first of the alternatives to DQn that we learned about, I would like to investigate the performance of the other methods described in the course -- specifically Dueling DQNs and the methods described in the RAINBOW paper.

I would also like to go back and examine further my Q-Network implementation to see how less or more parameters affect training time and ultimate performance.