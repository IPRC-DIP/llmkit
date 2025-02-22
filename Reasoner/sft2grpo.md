---
title: "From SFT to GRPO"
subject: reasoning model
license: CC-BY-4.0
keywords: datasets
date: 2025-02-22
authors:
  - name: Ziyuan Nan
    email: nanziyuan21s@ict.ac.cn
    affiliation: ICT CAS
---


## 背景介绍

在推理模型$\pi_{\theta}$的训练中，输入一个问题$q$，模型sample一组回答$G = \{o_{i}\}$，其中$i=1,2,3,...$。
对于每一个回答$o_i$，verifier会给出一个对应的奖励$r_{i}$。
我们暂认为：当回答正确时，$r_{i} = 1$, 否则$r_{i} = -1$。
我们的问题是：如何使用以上数据训练模型？

## SFT

首先我们观察一下SFT的梯度。
我们用 $o_{i,<t}$表示$o_{i}$的前$t$个token，$o_{i,t}$表示$o_{i}$的第$t$个token。
使用CrossEntropyLoss时，token $o_{i,t}$的loss为

$$
L_t(\theta) = - \log \pi_{\theta} (o_{i, t} | q, o_{i, <t})
$$

整个回答$o_i$的loss是各个token loss的均值。

$$
L_i(\theta) = - \frac{1}{|o_i|} \sum_{t} \log \pi_{\theta} (o_{i, t} | q, o_{i, <t})
$$

现在我们计算所有回答$G$的梯度。
在普通的Rejection Sampling中，所有错误回答被丢弃，只有正确回答参与训练，我们用`ReLU` 函数模拟这种情况。
并对所有回答的loss取均值。

```{math}
:label: sft_loss
L(\theta) = - \frac{1}{|G|} \sum_{i} \frac{1}{|o_i|} \sum_{t} \log \pi_{\theta} (o_{i, t} | q, o_{i, <t}) \mathop{ReLU} (r_i)
```

至此我们得到了一个iterative sft，或者用强化学习的术语，on-policy sft。

## 梯度上升 (Gradient Ascent)

很显然产生了一个想法：能不能让错误回答也参与训练？
对于正确的回答，应该采取梯度下降，对于错误的回答，梯度当然应该上升。这里可以与machine unlearning相联系。做法也很简单，只需要改变loss的符号。
对于[](#sft_loss)，我们只需要去掉`ReLU`，不截断错误回答的梯度。

```{math}
:label: reinforce
L(\theta) = - \frac{1}{|G|} \sum_{i} \frac{1}{|o_i|} \sum_{t} \log \pi_{\theta} (o_{i, t} | q, o_{i, <t}) \green{r_i}
```

到此，我们已经得到了一个强化学习算法：REINFORCE。但是REINFORCE方差大，还不够好。
我们先从reward开始继续优化。

## Reward Normalization and Clipping

推理模型训练里中的rule based reward的特点就是其分布是离散的，并不平滑。直接用离散的奖励训练模型可能会崩溃。
有什么方法可以平滑reward呢？当采样的回答足够多时，直接`normalization`就是一种简单有有效的方法。

我们记平滑后$o_{i}$的reward为$A_i$
$$
A_i = \frac{r_i - mean(r_1, r_2, ..., r_G)}{std(r_1, r_2, ..., r_G)}
$$

接下来，我们将通过例子观察平滑的作用。
在训练中，从问题中sample`1024`个回答，回答正确reward是1，回答错误reward是-1， 格式错误reward是-2。

1. 只有1个是正确回答，127个错误回答，896个格式错误。（训练早期）
2. 384个正确回答，512个错误回答，128个格式错误。（训练中期）
3. 896个正确回答，127个错误回答，1个格式错误。（训练后期）

以下表格是平滑后奖励的对比。

| 阶段 | 正确回答 | 错误回答 | 格式错误 |
|--|--|--|--|
| 非平滑(基线) | 1 | -1 | -2 |
| 训练早期 | 8.41 | 2.56 | -0.37 |
| 训练中期 | 1.43 | -0.45 | -1.38 |
| 训练后期 | 0.38 | -2.63 | -4.14 |

可以观察到，训练早期，偶尔出现的正确回答会被赋予极大的奖励。同理，训练后期偶尔出现的格式错误也会赋予极大的惩罚。
平滑可以帮助模型更好的训练偶尔的“灵光一闪”和错误。

对于reward，第二个技巧是clip，可以约束模型训练时的梯度大小，增强训练的稳定性。
$$
f^{\text{clip}}(x, A) = min(xA, clip(x, 1 - \epsilon, 1 + \epsilon)A)  
$$

将以上两个技巧加入到[](#reinforce)中：
```{math}
:label: reward_norm
L(\theta) = - \frac{1}{|G|} \sum_{i} \frac{1}{|o_i|} \sum_{t} \red{f^{\text{clip}}} ( \log \pi_{\theta} (o_{i, t} | q, o_{i, <t}), \red{A_i} )
```

最后，我们再从PPO里抄两个trick过来。

## Importance Sampling

由于是从PPO抄的，所以简单回顾一下。

我们想求$E_{x \sim p}[f(x)]$，但是我们又无法直接从分布$p$中采样，这时应该怎么办呢？
(在reasoning模型训练里，指数据是从上一个模型中采样的，但是优化目标是新模型)
方法是重要性采样：

```{math}
\begin{align*}
E_{x \sim p}[f(x)] &= \int f(x) p(x) dx \\
                   &= \int f(x) \frac{p(x)}{q(x)} q(x) dx \\
                   &= E_{x \sim q}[f(x) \frac{p(x)}{q(x)}] \\
\end{align*}
```

在[](#reward_norm)使用上式后，变为

```{math}
L(\theta) = - \frac{1}{|G|} \sum_{i} \frac{1}{|o_i|} \sum_{t} \red{f^{\text{clip}}} ( \frac{ \green{\cancel{\log}} \pi_{\theta} (o_{i, t} | q, o_{i, <t})}{ \green{\pi^{old}_{\theta} (o_{i, t} | q, o_{i, <t})}}, \red{A_i} )
```

证明可参考 [stack overflow](https://ai.stackexchange.com/questions/37958/where-does-the-proximal-policy-optimization-objectives-ratio-term-come-from)

## KL Penalty

KL惩罚的目的是让更新后的模型不要离原先的模型太远。
GRPO的做法是在loss中加入
$$
-\beta \mathbb{D} [ \pi_{\theta} || \pi^{ref}_{\theta}]
$$
至于GRPO的这一做法尚有异议；有效性也有待考证。

## GRPO

回顾以上，我们在on policy sft上加入了4个新的trick

1. 负梯度
2. 奖励平滑与裁剪
3. 重要性采样
4. KL惩罚

最终我们得到了GRPO。GRPO最novel的部分在于奖励平滑。
```{math}
:label: grpo_loss
L(\theta) = - \frac{1}{|G|} \sum_{i} \frac{1}{|o_i|} \sum_{t} \red{f^{\text{clip}}} ( \frac{ \pi_{\theta} (o_{i, t} | q, o_{i, <t})}{ \green{\pi^{old}_{\theta} (o_{i, t} | q, o_{i, <t})}}, \red{A_i} ) + \blue{\beta \mathbb{D} [ \pi_{\theta} || \pi^{ref}_{\theta}]}
```

根据之前DPO相关工作的经验，这4个trick中1和2比较重要，3和4可以尝试变种甚至删除。
对于未来工作，可能会出现以下几种可能：

- 放弃负梯度，用多训几个step替代负梯度的效果，就是sft-o1。
- 放弃重要性采样，走SimPO的路线，从正则的角度理解policy ratio。
- 放弃或者更改KL惩罚。当前系统中由2个模型，policy模型和ref模型。如果放弃KL惩罚，可以去掉ref模型，进一步简化RL系统。

