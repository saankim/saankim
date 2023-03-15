# Intro
순전파하지 않는 신경망; Hopfield network부터 Boltzman machine까지


# 홉필드 네트워크
![](hopfield-network.png)
1. Perceptron neuron은 $\pm 1$의 출력을 갖는다.
2. Fully connected Graph로 뉴런들이 연결되어 있다.
3. 뉴런간 Weight는 symmetric 하다.
결과적으로 신호를 주고받는 뉴런들은 flipping한다.

Hopfield의 연구에 따르면, flipping은 weight 변화를 거쳐 결국 어떤 패턴으로 optimize 된다.
- 에너지 관점
	flipping이 많이 일어나는 상태를 에너지가 높은 상태, flipping이 적게 일어나는 상태를 에너지가 낮은 상태로 보고 안정화되는 현상으로 해석할 수 있다.
- 기억 관점
	network가 한 번 안정화 되고 나면, 새로운 input이 주어지더라도 얼마간의 filipping을 거쳐 다시 안정한 상태를 회복하게 된다. 이를 weight에 의한 패턴의 기억이라고 해석할 수 있다. 뉴런 숫자에 따라서 표현할 수 있는 최종 패턴의 복잡도와 종류가 수학적으로 결정된다. 그리고 새로운 입력값은 어떤 최종 패턴 중 하나로 수렴한다.


## 기하적 직관
위 두 관점을 아우르는 시각화는 다음과 같다. 각각의 minimun이 하나하나의 기억이라고 해석할 수 있겠다.
![](energy-state-space.png)


## 신경과학 관점
아주 근접한 작은 minimun 두 개 중 하나를 지우거나, 충분히 깊은 minimum 하나를 강화하는 방향으로 뇌가 작용하는 결과가 꿈 이라는 해석이자 가설이 있다.


## 이후 연구
- Boltzmann machine (Geoffrey Hinton, 1983)
	에너지 최소화 기반 모델로 Hopfield network와 같은 관점을 공유한다.
- Hopfield Networks is All You Need
  홉필드 네트워크에서 뉴런의 활성화도로 $\pm 1$ 만을 사용하지 않고 continuous한 값을 사용하게 되면 기억 가능한 값의 지점이 exporential 하게 증가하는 현상에 대한 연구
- Attention, Attention is All You Need
  Hopefiled 네트워크와 attetion의 식이 구조적으로 같다.
- 당시 energy 기반 모델이 주목받은 이유는 Ising model과 같이, molecule들의 dipole이 field에 의해 flip 되는지를 복잡계적으로 접근한 연구들의 영향이 있다. 영상 55분 45초 경.


# 볼츠만 머신


# References
- John Hopfield, 1982