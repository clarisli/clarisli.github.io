---
layout: post
title:  "DQN: Deep Double Q-Learning"
date:   2019-07-01 10:14:00 +0800
categories: [AI]
disqus: true
---

DQN overestimates Q values. The agent values an action much higher than its true value. These estimation errors may lead to longer learning time and poor policies. [**Double DQN**](https://arxiv.org/abs/1509.06461) is the algorithm combining DQN and [**double Q-learning**](https://dl.acm.org/citation.cfm?id=2997187) to reduce the overestimations. 

Before you continue reading, I recommend you read my [previous post on DQN](https://clarisli.github.io/ai/2019/06/13/deep-rl-dqn.html).

### Probability Basics for Reinforcement Learning

It's important to understand we are dealing with uncertainties. The randomness can be from the environment, reward functions, or even the policies. For example, a robot tries to move right, but the floor can sometimes be slippery, therefore sometimes it ends up at left. 

Reinforcement learning uses [**expectations**](https://en.wikipedia.org/wiki/Expected_value) to address the randomness. If you are not familiar, expectation is the mean of a random variable. Take for instance, when you roll the dice with an expected value of 3.5 for many times, the mean of the outcomes will be very close to 3.5. A [**random variable**](https://www.statisticshowto.datasciencecentral.com/random-variable/) is a placeholder for a value to be sampled from a probability distribution. For example, rolling the dice is taking a sample from the dice’s probability distribution. 

Q function is a random variable with a unknown distribution, mapping the state-action pairs into accumulated future rewards. The goal of a Q function is to predict the consequences of an action *a* at a state *s*. It’s the **return** - expected accumulated discounted future rewards - given starting in the state *s*, taking the action *a* and following the policy $$ \pi $$.

$$ Q_\pi(s, a) = \mathbf{E} [R_1 + \gamma R_2 … | S_0 = s, A_0 = a, \pi ] $$

Note the expectation is used here. Often we only care about the mean of a random variable but not its individual samples. Estimating a Q value is equivalent to sampling from the distribution. In reinforcement learning, we are trying to model the underlying distribution with the samples we take from it. 

### Understand Overestimation

Now we know we're doing estimations, but why do we overestimate? 

Let’s look into how Q-learning update at each step:

$$Q(s, a) = Q(s, a) + \alpha (R(s, a, s') + \gamma \max_a Q(s', a') - Q(s, a))$$

Pay attention to $$ \max_a Q^*(s', a')$$, where we sample a Q value for each action at the state, and take a maximization over them. The overestimations take place here.

Generally, it’s a bad idea to take maximization over samples, because little estimation errors can lead to overestimation. To see the reasons behind, take a dice as an example. Think of rolling a dice as sampling the Q value for an action. The dice has an expected value of 3.5. You roll the dice 100 times and take the maximization over all outcomes. It is very unlikely you will get a 3.5. Instead, you will probably get a value much higher - because one sample of 6 is sufficient to bias the result to 6. This effect is known as the **Optimizer’s Curse**.

The curse doesn’t just end here. In standard Q-learning, the same Q values are used to choose the best next action and determine the value of that action. This leads to a double curse, where an overestimation builds upon another overestimation. The action with an overestimated value is likely to be selected as the next action. Then, the same overestimated value is used as the target to update the Q function. Later when we visit the same state, the previously overestimated action is likely to get selected again. Because the Q function has been updated towards an overestimated value, it's likely to return a high value for the action. This overestimation propagates throughout the learning.

It shouldn't be a problem if we overestimate every action equally. In maximization only the difference between values matters. Take for instance four actions with true values 1, 2, 3, 4 and the agent overestimates equally with values 3, 4, 5, 6, the fourth action will still be selected as the next action. However, when we only overestimate some actions, the agent may waste a lot of time exploring the wrong actions, while missing those truely valuable ones. For example, if the agent only overestimates the first action above with values 10, 2, 3, 4, then the agent will select the first action instead. The learning slows down, or even worse, the agent fails to find an optimal policy.

Eliminating the estimation errors is a straighforward solution. But in reality, it's not an option. In estimation we are trying to find the initially unknown true values with limited information. Some levels of inaccuracies will always exist during learning. Estimation errors are unavoidable.

### Double DQN

If overestimations can’t be removed, can it be reduced?

**Double Q-learning** provides an answer to this question. With a goal to reduce overestimations, two Q functions are used to decouple the action’s selection and evaluation. At each step, samples are assigned randomly to one of the Q functions. While the Q function gets assigned learns online and selects the next action, another Q function evaluates the value. This works because it’s unlikely both Q functions overestimate the same action at the same time.

$$Q_1(s, a) = Q_1(s, a) + \alpha (R(s, a, s') + \gamma Q_2(s’, \max_a Q_1(s', a')) - Q_1(s, a)))$$
$$Q_2(s, a) = Q_2(s, a) + \alpha (R(s, a, s') + \gamma Q_1(s’, \max_a Q_2(s', a')) - Q_2(s, a)))$$

In first formula $$Q_1$$ gets the sample, and in the second $$Q_2$$ gets the sample.

The same concept can be applied to DQN to reduce overestimations. We call this algorithm **Double DQN**, where the target network in DQN is used to select action while the online network does the evaluation.

$$Q_\theta(s, a) = Q_\theta(s, a) + \alpha (R(s, a, s') + \gamma Q_\theta(s’, \max_a Q_{\theta^{-}}(s', a')) - Q_\theta(s, a)))$$


### References

[1] van Hasselt, H., Guez, A., & Silver, D. (2015). [Deep Reinforcement Learning with Double Q-learning.](https://arxiv.org/abs/1509.06461)
