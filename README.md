# Reinforcement Learning with OpenAI Gym: CartPole Environment

This repository contains two Jupyter Notebook files (cartpole-DQN.ipynb and cartpole-q-learning.ipynb) that explore two popular reinforcement learning algorithms for solving the CartPole control problem using OpenAI Gym.

## About the CartPole Environment:

The CartPole is a classic benchmark environment in reinforcement learning. It simulates a pole attached to a cart by a hinge. The agent's goal is to learn to push the cart left or right to keep the pole upright for as long as possible.

## Jupyter Notebooks:

### cartpole-DQN.ipynb 
This notebook implements a Deep Q-Network (DQN) agent to solve the CartPole problem. DQN is a powerful technique that combines Q-learning with a neural network to handle large and continuous state spaces.
### cartpole-q-learning.ipynb
This notebook implements a traditional Q-learning agent for the CartPole task. Q-learning is a model-free reinforcement learning algorithm that learns a policy by directly interacting with the environment.

## What's Included:

Each notebook provides a comprehensive exploration of the chosen algorithm, including:

* Importing necessary libraries (OpenAI Gym, NumPy, etc.)
* Setting up the CartPole environment
* Discretizing the continuous state space (for Q-learning)
* Implementing the core algorithm (DQN or Q-learning) with detailed explanations
* Defining functions for exploration (epsilon-greedy strategy) and learning rate decay
* Training the agent over multiple episodes
* Evaluating the agent's performance and visualizing results (optional)

## Getting Started:

Install Required Libraries: Ensure you have Python, Jupyter Notebook, and the necessary libraries (Gym, NumPy, etc.) installed on your system. You can use pip install to install them.
Open the Notebooks: Launch Jupyter Notebook and open the desired notebook (cart-pole-DQN.ipynb or cart-pole-qlearning.ipynb).
Run the Cells: Execute the cells in the notebook one by one to follow the code and train the agent.
 
## Concepts Utilised:

By working through these notebooks, we have gained a solid understanding of:

* Reinforcement learning concepts: state, action, reward, discount factor, policy
* Q-learning algorithm and its implementation
* Deep Q-Networks (DQN) and their application in continuous state spaces
* Discretization techniques for Q-learning
* Epsilon-greedy exploration strategy
* Training and evaluating reinforcement learning agents
