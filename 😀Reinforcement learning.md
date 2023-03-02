# 강화학습


1. Trajectory, states and their transition
	- $\tau=\left(s_0, a_0, s_1, a_1, \ldots\right)$
	- $s_{t+1}=f\left(s_t, a_t\right)$
	- $s_{t+1} \sim P\left(\cdot \mid s_t, a_t\right)$
2. Return: Sum of discounted rewards in the future
	- $R_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\cdots=\sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$
	- $r_t=R\left(s_t, a_t, s_{t+1}\right)$
	- $J(\pi)=\int_\tau P(\tau \mid \pi) R(\tau)=\underset{\tau \sim \pi}{\mathrm{E}}[R(\tau)]$
		-  $P(\tau \mid \pi)=\rho_0\left(s_0\right) \prod_{t=0}^{T-1} P\left(s_{t+1} \mid s_t, a_t\right) \pi\left(a_t \mid s_t\right)$
	- $\pi^*=\arg \max _\pi J(\pi)$
3.  Value (on-policy): Expected return if the agent starts in state s and always act according to policy $\pi$
	- $V^\pi(s)=\underset{\tau \sim \pi}{\mathrm{E}}\left[R(\tau) \mid s_0=s\right]$
	- with optimal policy,
	  $V^*(s)=\max _\pi \underset{\tau \sim \pi}{\mathrm{E}}\left[R(\tau) \mid s_0=s\right]$
4. Q function (on-policy action-value function): Expected return if starting in state $s$, take an arbitrary action $a$ (which may not have come from the policy), and ten forever after act according to policy $\pi$
	- $Q^\pi(s, a)=\underset{\tau \sim \pi}{\mathrm{E}}\left[R(\tau) \mid s_0=s, a_0=a\right]$
	- with optimal policy, 
	  $Q^*(s, a)=\max _\pi \underset{\tau \sim \pi}{\mathrm{E}}\left[R(\tau) \mid s_0=s, a_0=a\right]$



# Reinforcement Learning an Introduction 
#TODO 
# 강화학습이론

## 딜레마
### 1. exploration vs. exploit

### 2. catastrophic forgeting: stability vs. plasticity

이 두 딜레마에 대해서 AI, emotion, internal state를 이용해서 업데이트 할 수 있지 않을까?
⇒ task가 어렵고 복잡해질때, 더 낮은 복잡도의 agent도 interal state와 emotion을 anker로서 삼고 주어진 일을 더 잘 수 행할 수 있을 것이라고 생각함.


인간이 볼 때 어려운 문제 바둑이나 게임은 오히려 계산지능에겐 풀기 쉬운 문제일 수 있다. 그런데 value updating이 필요한, 밥먹기 걷기 같은 활동이 오히려 더 어려울 수 있다.


## 자료
### RL 책
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiVs5yOvMH6AhUOBaYKHbbyCQoQFnoECAcQAQ&url=https%3A%2F%2Fweb.stanford.edu%2Fclass%2Fpsych209%2FReadings%2FSuttonBartoIPRLBook2ndEd.pdf&usg=AOvVaw3bKK-Y_1kf6XQVwR-UYrBY


### Introduction to Reinforcement Learning with David Silver
https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver



# References
- https://mattmazur.com/2015/03/17/
- https://spinningup.openai.com/en/latest/spinningup/rl_intro.html/
- https://towardsdatascience.com/intro-to-reinforcement-learning-temporal-difference-learning-sarsa-vs-q-learning-8b4184bb4978
- https://www.cs.hhu.de/fileadmin/redaktion/Fakultaeten/Mathematisch-Naturwissenschaftliche_Fakultaet/Informatik/Dialog_Systems_and_Machine_Learning/Lectures_RL/L3.pdf