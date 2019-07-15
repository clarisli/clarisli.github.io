---
layout: post
title:  "DQN: Prioritized Experience Replay"
date:   2019-07-07 14:21:00 +0800
categories: [AI]
disqus: true
---

Experience replay allows agents to remember and reuse past experiences. It samples the experiences uniformly from the memory. However, some experiences are more important than others and should not be treated equally. With [**prioritized experience replay (PER)**](https://arxiv.org/abs/1511.05952), the agents replay important experiences more often and therefore, learn more efficiently.


Why are we using experience replay in the first place? Without experience replay, the samples taken from reinforcement learning are highly correlated and breaking the [i.i.d. assumption](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) - a very common assumption in many machine learning algorithms that the samples are independent and identically distributed random variables. In addition, without memory, the agent forgets experiences immediately. Instead of waiting for important but rare experiences to happen again, which might take forever, it seems wiser to remember them.

### Greedy TD-Error Prioritization

Prioritized experience replay takes experience replay one step further. Experiences are not only remembered but also replayed according to importance. The more available to learn from an experience, the more important it is, and the more frequent we want to replay it. We use the difference between the target and prediction to determine the level of importance. The greater the difference, the greater the room for improvement, and the greater the importance. This difference is called the TD error. 

$$ \delta = \overbrace{R(s, a, s') + \gamma \max_a Q(s', a')}^{\text{target}} - \overbrace{Q(s, a)}^{\text{prediction}} $$


**Greedy TD-error prioritization** is an algorithm that stores each experience along with its TD error. It’s called *“greedy”* because the largest TD error experience always gets to replay.  New experiences are assigned with the largest TD error to make sure they are replayed for at least once.

However, this approach has some issues. The system overfits when there are experiences with large initial TD errors. During training, step by step, similar to gradient descent, the TD errors shrink slowly. The system trains with the same experience for many steps, until it finds another experience with a larger error. When a system trains with same experiences for too many times, it overfits. On the other hand, an experience with small initial TD error has a problem too. It has to wait for a long time before gettin replayed, or may even never get replayed when there's a sliding window. 

Note the system is very sensitive to stochastic rewards. For example, when an unimportant experience gets a high initial TD error, because of the randomness in rewards, the agent will waste a lot of time learning from it repeatedly.

### Stochastic Prioritization

We want to make sure every experience gets a chance to be picked, and the higher the priority, the greater the probability. **Stochastic prioritization** is a sampling method created for this purpose. The probability for experience $$i$$ to be picked is defined as

$$ P(i) = \frac{p_i^\alpha}{(\sum_k{p_k^\alpha})} $$

$$p_i$$ is the priority of experience $$i$$. The priority can be defined in various ways. Either directly relates to TD error

$$ p_i = |\delta_i| + \epsilon $$

$$\epsilon$$ is a constant to ensure the experience with zero error still stands a chance to be picked. 

or indirectly through the ranks 

$$ p_i = \frac{1}{rank(i)}$$

where the $$rank(i)$$ is the rank of the experience $$i$$ when the replay memory is sorted by TD errors. 

The hyperparameter $$\alpha$$ controls the randomness involved. Set $$\alpha = 0$$ to do uniform random sampling, and $$\alpha = 1$$ to do greedy prioritization.

### Importance Sampling 

Prioritize sampling introduces a bias toward high-priority experiences. To understand, let’s review how Q-learning update at each step:

$$Q(s, a) = \overbrace{Q(s, a)}^{\text{estimations}} + \alpha (\overbrace{R(s, a, s')}^{\text{samples}} + \gamma \max_a \overbrace{Q(s', a')}^{\text{estimations}} - \overbrace{Q(s, a)}^{\text{estimations}})$$

In general, the samples and estimations should fall in the same distribution. When they don’t, the solution we find will be wrong. When we assign higher probabilities to high-priority experiences, we are changing the underlying sampling distribution. The high-priority experiences are been picked up more often than they should under the true distribution. To compensate this, we use **importance sampling** weights:

$$ w_i = (\frac{1}{N} \cdot \frac{1}{P(i)})^\beta $$

During Q-learning update, we use $$w_i\delta_i$$ instead of $$\delta_i$$ to scale the TD error. 

The hyperparameter $$\beta$$ controls the level of compensation. When $$\beta = 1$$, the prioritize sampling probabilities are fully compensated. In reinforcement learning, unbiased updates are most important near convergence at the end of the training. We typically set $$\beta$$ close to zero at the beginning of learning, and anneal it up to one over the training. 


### References

[1] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952). 

