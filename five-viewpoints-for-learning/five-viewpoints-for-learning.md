# Five Viewpoints for learning
인공지능과 뇌과학 분야에서 제안되어온 학습 기제에는 크게 아래와 같은 분류들이 있다.
1. Supervised vs. Unsupervised vs. Self-supervised
   지도 vs. 비지도 vs. 자가지도
2. Discriminative(classification) vs. Generative
   구분 vs. 생성
3. Model-free vs. Model-based
   모델이 없는 vs. 모델에 기반한
4. Exploitation(pragmatic) vs. Exploration(epistemic)
   활용 vs. 탐험
5. Nature(inductive bias) vs. Nuture(inference and learning)
   본성(내재된 편향) vs. 양육(추론과 학습)
   

## Supervised vs. Unsupervised vs. Self-supervised
뇌가 학습을 하는 과정에서 각각의 경우에 대해 explicit reward(negative errror)가 바로 주어지지는 않는다. 따라서 뇌의 학습기제는 비지도학습에 가까울 것이라는게 현재의 컨센서스다. 다만, 현대 들어서는 self-supervised 방식일 것으로 기대하는 접근들도 많다.


## Discriminative(classification) vs. Generative
- Discriminative 모델은 boundary를 학습해서 입력 데이터를 분류한다.
- Gernerative 모델은 internal representation을 학습해서 목적하는 분포와 유사하게 분포된 데이터를 생성

수식으로 쓰자면 각각 아래 확률을 학습한다. 데이터를 $X$, 라벨을 $Y$로 표현했다.
- Discriminative $P(Y \mid X=x)$
- Generative $P(X, Y)$
	- Conditional generative $P(X \mid Y = y)$

[Computational Cognitive Neuro Science 2021 conference](https://www.youtube.com/watch?v=bEG4T18le5g&ab_channel=CognitiveComputationalNeuroscience)에서 이 논의가 진행되었다.


## Model-free vs. Model-based
강화학습 관점에서, 환경의 transition을 직접 예측할 수 있는 방법을 가지고 있으면 model-based로 볼 수 있다. 특히 RL관점에서는 아래 그림처럼 분류할 수 있다. 주의할 점은 policy는 model이 아니라는 점이다. policy가 아니라 환경에 대한 simulative model을 가지고 있느냐로 구분해야 한다.
![](rl-categories.png)


## Exploitation(pragmatic) vs. Exploration(epistemic)
활용만 한다면 새로운 정보를 얻고 더 큰 reward가 기대되는 행동을 개발하기 어렵다. 반면 탐험을 하면 더 탐험성 높은 탐색을 위해 reward를 어느정도 포기하는 결정을 해야한다. 지능은 이 둘을 적절히 종류하면서 reward를 최대화하고자 한다.


## Nature(inductive bias) vs. Nuture(inference and learning)
뇌과학 분야에서는 피아제의 발달 5단계를 비롯한 개념들로 연구되어 왔다. 인공지능 분야에서는 inductive bias라는 표현이 흔히 사용된다.

참고자료
- [Building Machines That Learn and Think Like People](https://arxiv.org/pdf/1604.00289.pdf)
- [Towards developmental AI](https://www.youtube.com/watch?v=s9EeozO6fp8)


# Reference
- https://www.youtube.com/watch?v=vLfHhjxeXrY
- https://www.youtube.com/watch?v=bEG4T18le5g&ab_channel=CognitiveComputationalNeuroscience

