# Bayesian optimization
# 베이지안 최적화
베이지안 최적화는 베이즈 정리를 바탕으로 글로벌 최적화를 위한 Sequential optimization하는 알고리즘이다. 베이지안 최적화를 통해 풀 수 있는 문제를 수식으로 표현하면 $\max_{x \in A} f(x)$ 와 같다. 보통 20차원 이하의 차원에 대해서 잘 동작하며, 이 이상의 차원에 대해서는 Curse of dimensionality를 겪게 된다. 베이지안 최적화 과정에서 차원의 저주는 가우시안 프로세스에서 correlation 계산시 고차원의 점들간의 correlation이 비슷해지는 문제에서 기인한다.


## 알고리즘
![베이지안 최적화 과정 그림](https://upload.wikimedia.org/wikipedia/commons/0/02/GpParBayesAnimationSmall.gif)
1. 아무 function으로 회귀
   앞으로 이 함수를 posterior function이라 부르겠음.
   이 이름은 베이지안 법칙으로 부터 오는 이름.
2. 회귀한 function으로부터, 기대값이 최고인 정의역벡터 $\mathrm{x}$를 구함
3. 해당 정의역벡터를 입력으로 하는 실험을 진행해서, $f_{true}(\mathrm{x}) = \mathrm{y}$ 를 구함
4. $\mathrm{(x, y)}$를 추가해 회귀한 posterior에 대해 반복
	1. 추가해 회귀할때 correlation function과 커널에 따라 이웃한 점들의 posterior를 새 데이터를 prior로 보고 베이지안 법칙에 따라 계산하게 됨.
이 과정에서, posterior function으로 보통 가우시안 프로세스회귀 를 사용한다. 특별히, 가우시안 프로세스 회귀를 surrogate model로 사용한 베이지안 최적화 과정을 Kriging이라고도 한다.


## 일반적인 구현
가장 일반적인 형태의 베이지안 최적화 구현은 가우시안 프로세스에 acquisition function이 추가된 형태를 가지고 있다. 


### Acquisition function
Acquisition function은 베이지안 최적화 과정에 따라 가정된 posterior 함수를 입력받아, 다음 실험을 위한 하나 또는 여러 정의역 벡터 $\mathrm{x}$를 선택하는 함수다. 다른 모든 Optimization 알고리즘과 유사하게, 탐색/최적화 사이의 trade-off 관계에 대해 어떤 목적에 더 중점을 두고 탐색해나갈지에 따라 다양한 acquisition function이 있을 수 있다.
참고로, 아래와 같은 변수들을 조정했을때도 탐색과 최적화 중에 한 쪽에 더 중점을 두도록 모델을 설계할 수 있다.
- Posterior 함수의 분산행렬 $\Sigma$를 생성하는 correlation function의 선정
	- 더 멀리 있는 $\mathrm{x}$도 더 비슷한 correlation을 가지도록 하면 모델이 탐색을 덜 수행하도록 만들 수 있다.
- Posterior 함수의 분산 $\sigma$
	- 분산을 크게 둘 수록 더 많은 지점을 탐색하려는 것 처럼 행동하기 쉽다.
- Posterior 함수의 error 범위
	- 에러 범위를 크게 둘 수록 더 많은 지점을 탐색하면서 최적값을 찾아나가게 된다.

#### 대표적인 acquisition function
- Expected Improvement, EI


## 베이즈 정리와의 연관성
베이지안 최적화 과정을 posterior distribution을 prior $\mathrm{(x,y)}$가 추가됨에 따라 업데이트 해 나가는 과정으로 해석한다면, prior를 통해 posterior를 계산할 수 있는 베이즈 정리로부터 유도될 수 있음을 알 수 있다.
- 베이즈 정리
	- 주어진 값: prior
	- 찾고자 하는 값: posterior
- 베이지안 최적화
	- 주어진 값: 데이터셋
	- 찾고자 하는 값: correlation/kernel function에 따라 결정된 이웃 정의역 $\mathrm{x}$의 posterior


### 무모수 모델인가?
베이즈 정리와의 연관성을 바탕으로 베이지안 최적화를 무모수 모델로 생각할 수 있다. 결론부터 말하자면 베이지안 최적화 과정은 무모수 모델이 맞다. 하지만 베이지안 최적화 과정에서 posterior 함수로 흔히 사용하는 GP regressor는 hierarchy prior를 사용하지 않으면, mean function에서 모수추정을 하고 있기 때문에 베이지안 최적화 모델 전체가 무모수 모델이라 볼 수는 없다.


#### 무모수 모델이 되는 법
더 자유로운 실험값에 대한 optimization을 위해 무모수 모델에 더욱 가깝게 만들 수 있는 방법이 있다.


##### GP에서 hierarchy prior를 사용하는 방법
GP mean function에서 가정하는 모수추정모델을 두지 않을 수 있는 방법이다. Hierarchy prior로 prior를 chainging 한 후 mean function 선정에 관여하는 hyperparameter(mean function polynomial의 계수 등)에 대해 정리하면 mean function과 그 파라미터를 error(robustness) $\epsilon$에 대한 함수로 둘 수 있다.


##### RBF 커널과 중심극한정리를 이용한 방법
Gaussian process 등 posterior function에 Radial basis fuction kernel (RBF kernel)을 적용해주면, Central limit theorem에 따라 모수를 가정하는 모델에 의한 최적화 프로세스의 왜곡을 줄일 수 있다는 해석이 있으나, 이는 여기서 증명하지 않겠다.


## 구현
특히 Machine learning 또는 Neural network모델에서 하이퍼파라미터 최적화를 위해 사용할 수 있다.



# References
- https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f
- https://machinelearningmastery.com/bayes-theorem-for-machine-learning/
- P. I. Frazier, “A Tutorial on Bayesian Optimization,” arXiv, arXiv:1807.02811, Jul. 2018. doi: [10.48550/arXiv.1807.02811](https://doi.org/10.48550/arXiv.1807.02811).
- J. Brownlee, “How to Implement Bayesian Optimization from Scratch in Python,” _Machine Learning Mastery_, Oct. 08, 2019. [https://machinelearningmastery.com/what-is-bayesian-optimization/](https://machinelearningmastery.com/what-is-bayesian-optimization/) (accessed Aug. 31, 2022).
- [GpyOpt](https://sheffieldml.github.io/GPyOpt/)