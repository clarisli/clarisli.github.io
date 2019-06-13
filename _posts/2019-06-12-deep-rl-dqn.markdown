---
layout: post
title:  "Deep RL: Deep Q-Network DQN"
date:   2019-06-13 16:29:00 +0800
categories: [AI]
disqus: true
---

Can machines play video games like a human? 

I want to create a robust agent good at many games. It learns directly from what it “sees” - just like how humans learn to play the video games. 

In 2015, following the success of deep learning in computer vision, DeepMind published an innovative method called [Deep Q-Networks (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) combining deep learning with reinforcement learning (RL) to train such an agent. This agent learns directly from a video input of Atari games with a convolutional neural network and a modified Q-learning algorithm. In this experiement, DQN outperforms all previous approaches on six games and a human expert on three games.

### Background

To get started, let’s define some basic terms in RL and build the knowledge from there.

Imagine there’s an **agent** in an **environment**. The agent is in a **state**, it takes an **action** to interact with the environment. For each action it takes, the environment responds with a **reward**. There’s no fixed format of a state, it is designed according to the specific goal we want the agent to achieve. For example, the state for a robot in a 2D grid world can be defined as a coordinate of (x, y), there can be four actions available {left, right, top, down}, and the reward can be an integer of {-1, 0, 1}.

Reinforcement learning's goal is to find a sequence of actions that will take the agent from a state to a reward. At each state s, the agent gets to choose an action to take, then it receives a reward r, and ends up at the next state s’. This (s, a, r, s') is called a **transition**:

$$s \xrightarrow{a} r, s'$$

The way an agent chooses an action at each state is called a **policy**. Ultimately, we want to learn an **optimal policy** that leads to the largest total rewards.

In Q-learning, the goal is to learn the Q function, **Q(s, a)**. Q returns the total rewards for taking the action a at the state s. Note that it's the total rewards of a sequence of states and actions: (s0, a0, r0, s1, a1, r1, s2, a2, r2,...), not only the reward received at the current (s0, a0). 

Suppose there's a function **Q\*(s, a)** that returns the total rewards following an optimal policy. Then getting the largest rewards becomes an easy task, the agent can simply follow a **greedy policy**, choosing the action that maximizes Q\* at each state. But the issue is we don't know the groud truth Q\*.   

But we can approximate Q\* instead. Depends on the complexity, the approximation function for Q\* can be in many forms. As simple as a lookup table, or as complex as a deep neural network. For video games, we will use a convolutional neural network that takes in images as input and outputs an estimation of Q value for each action. The goal of this network is to learn Q\*.

Neural networks with more than one hidden layers are called a “deep” neural network. This is where the names of Deep RL and Deep Q-learning (DQN) come from.

Note that not all rewards valued the same. Similar to investment, we value the reward we receive now more than the rewards in the future. The further away the rewards are, the lower values they hold. We use a **discount factor γ** to reflect this concept in Q function:

$$Q^*(s, a) = r_0 + \gamma r_1 + \gamma^2 r_2 + \gamma^3 r_3 + ...$$

With this, we can rewrite the formula in the recursive form a.k.a the famous **Bellman equation**:

$$Q^*(s, a) 
= r_0 + \gamma (r_1 + \gamma r_2 + \gamma^2 r_3 + ...) 
= r_0 + \gamma \max_a Q^*(s', a)$$

This is the heart of Q-learning. The agent learns **online** from the immediate reward it receives at each transition as it explores. The Q approximation gets closer to Q* at each transition and eventually converges to Q*. 

### Challenges

Q-learning does not work out of the box with deep learning. In deep learning, we often work with large labelled dataset, independent samples with fixed distribution, where in RL the samples are correlated and the distribution is non-stationary. These differences may cause the network to diverge.

Samples are highly correlated in RL because we collect them from consecutive states. For example, if we use video frames as states, the current frame is correlated to its previous frame and its next frame. A neural network trained with highly correlated samples may overfit and fail to converge. 

Another issue is that the network is chasing a moving target. We are training the network to find the ground truth Q* with a non-stationary, estimated target Q. Target Q is unstable because we update it at each step with the latest data sample. The network’s current parameters determines Q, and Q determines the next data sample, then the data sample is used to train and update the parameters. 

### Solutions
A technique called **experience replay** is used to overcome these challenges. The main concept is to store the transitions, then at each training step, randomly sample transitions from this memory. In this way, we break the correlation between samples and the dataset distribution becomes more stable. The data efficiency is also increased as a transition can be reused many times before we throwing it away.
 
Training with **off-policy** will fix the moving target issue. The idea is to have two networks, one learns new parameters with the samples generated by another one. The network used for samples has its target fixed, and in this way, we break the correlation loop between samples and parameters. 

### Implementation

I will implement the DQN algorithm with [TensorFlow](https://www.tensorflow.org/) and [OpenAI Gym](https://gym.openai.com/) in another post.

