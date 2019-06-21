---
layout: post
title:  "Deep RL: Deep Q-Network DQN"
date:   2019-06-13 16:29:00 +0800
categories: [AI]
disqus: true
---

Can machines play video games like a human? 

A group of researchers created a robust agent good at many games. The agent learns directly from what it “sees” - just like how humans learn to play the video games. 

In 2013, following the success of [deep learning](http://ufldl.stanford.edu/tutorial/) in computer vision, DeepMind published an innovative method called [Deep Q-Networks (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) combining deep learning with [reinforcement learning (RL)](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to train such an agent. This agent learns directly from a video input of Atari games with a convolutional neural network and a modified Q-learning algorithm. In their experiement, DQN outperforms all previous approaches on six games and a human expert on three games.

### RL Basics

To get started, let’s define some basic terms in RL and build the knowledge from there.

In reinforcement learning, there’s an **agent** interacting with an **environment** to learn control **policies**. At each timestep t, the agent observes a **state s** and takes an **action a** to interact with the environment. The action determines the **reward r** that the agent gets from the environment and the **next state s'**. 

There’s no fixed format of a state, it is designed according to the specific goal we want the agent to achieve. For example, the state for a robot in a 2D grid world can be defined as a coordinate of (x, y), there can be four actions available {left, right, top, down}, and the reward can be an integer of {-1, 0, 1}.

Reinforcement learning's goal is to find a sequence of actions that will take the agent from a state to a reward. At each state, the agent gets to choose an action to take, then it receives a reward, and ends up at the next state. This 4-tuple (s, a, r, s') is called a **transition** or **experience**:

$$s \xrightarrow{a} r, s'$$

How an agent chooses an action at each state is called a **policy**. Ultimately, we want to learn an **optimal policy** that leads to the largest total rewards.

### Q-Learning

In Q-learning, we learn a Q function, **Q(s, a)**, to implicitly learn an optimal policy. Q function returns the total rewards for taking the action a at the state s. Note that it's the total rewards of a sequence of states and actions: (s0, a0, r0, s1, a1, r1, s2, a2, r2,...), not only the reward received at the current (s0, a0). The higher the Q value, the better the action.

If we know a function **Q\*(s, a)** that returns the total rewards following an optimal policy. Then getting the largest rewards is easy, the agent can simply follow a **greedy policy**, choosing the action that maximizes Q\* at each state. 

However, we don't know Q\*. We find Q\* through approximation. The goal is to get an estimate very close to the ground truth Q\*. We do this by iteratively updating the Q-value estimate towards the ground truth. Similar to gradient descent, where we iteratively move toward the steepest downward direction at each step until reaching the minimum. 

Now, let's take a closer look at what's happening in a single step. Before an agent acts, it uses Q(s,a) to estimate the rewards for each action available at state s. Then the agent picks an action to act and observes a reward. We can then combine this observed reward with the maximum Q-value estimate for the next state s' to create a one-step look ahead:

$$target = R(s,a,s') + \gamma \max_a Q^*(s', a')$$

We use this value as the target. More information is gained through the observed reward, making it a better estimate than Q(s, a). 

Note that not all rewards valued the same. Similar to investment, we value the reward receiving now more than the rewards in the future. The further away the rewards are, the lower values they hold. The **discount factor γ** is used to reflect this concept in Q function:

$$Q^*(s, a) = r_0 + \gamma r_1 + \gamma^2 r_2 + \gamma^3 r_3 + ...$$

We often set the discount factor with a value less than 1. With this, we can rewrite the formula in the recursive form a.k.a the famous **Bellman equation**:

$$Q^*(s, a) 
= r_0 + \gamma (r_1 + \gamma r_2 + \gamma^2 r_3 + ...) 
= r_0 + \gamma \max_a Q^*(s', a)$$

This target guides Q(s, a) moving toward the ground truth:

$$Q(s, a) = Q(s, a) + \alpha (R(s,a,s') + \gamma \max_a Q^*(s', a') - Q(s, a))$$

This is the essence of Q-learning. The agent learns **online** from the immediate reward it receives at each transition as it explores. Iteratively, the Q approximation gets closer to Q\* at each transition and eventually converges to Q\*. 



### Deep Q-Learning

Depends on the complexity, the approximation function for Q\* can be as simple as a lookup table, or as complex as a deep neural network. Lookup table cannot keep track of the many unique states in Atari games. Instead, we use a convolutional neural network. It takes in images as input and outputs an estimation of Q value for each action. 

The goal of this network is to learn Q\*. Instead of updating Q(s,a) as with a lookup table, we update the network's parameters to minimize the loss function:

$$L(s,a|\theta) = (r + \gamma \max_a Q^*(s', a'|\theta) - Q^*(s, a|\theta))^2$$

Neural networks with more than one hidden layers are called a “deep” neural network. This is where the names of Deep RL and Deep Q-learning (DQN) come from.

### Challenges

Q-learning does not work out of the box with deep learning. In deep learning, we often work with large labelled dataset, independent samples with fixed distribution, where in RL the samples are correlated and the distribution is non-stationary. These differences may cause the network to diverge.

Samples are highly correlated in RL because we collect them from consecutive states. For example, if we use video frames as states, the current frame is correlated to its previous frame and its next frame. A neural network trained with highly correlated samples may overfit and fail to converge. 

Another issue is that the network is chasing a moving target. We are training the network to find the ground truth Q* with a non-stationary, estimated target Q. Target Q is unstable because we update it at each step with the latest data sample. The network’s current parameters determines Q, and Q determines the next data sample, then the data sample is used to train and update the parameters. 

### Solutions
A technique called **experience replay** is used to overcome these challenges. The main concept is to store the transitions, then at each training step, randomly sample transitions from this memory. In this way, we break the correlation between samples and the dataset distribution becomes more stable. The data efficiency is also increased as a transition can be reused many times before we throwing it away.
 
Training with **off-policy** will fix the moving target issue. The idea is to have two networks, one learns new parameters with the samples generated by another one. The network used for samples has its target fixed, and in this way, we break the correlation loop between samples and parameters. 

### Implementation

I will implement the DQN algorithm with [TensorFlow](https://www.tensorflow.org/) and [OpenAI Gym](https://gym.openai.com/) in another post.

