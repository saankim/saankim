# Random process
Stochastic process, 확률과정, 확률변수의 수열, 확률변수의 집합

이름이 어려운데, 어려운 개념은 아니다. 예시를 통해 직관적 이해부터 시작해보자.
- 주사위를 던져보자.
- 계속 던지면서, 매번 무슨 수가 나오는지 기록해두자.
- 축하한다! 당신은 방금 확률과정을 수행했다!

곧, 확률과정은 어떠한 확률분포를 따르는 사건이 시간적으로 연속해 발생한 것을 가리키는 말이다.


## 수학적 형식화
수학적 형식화를 조금 추가해 좀 더 엄밀한 정의와 이해를 도모해보자. 확률분포를 따르는 사건은 확률변수라고 부르자. 결국 확률과정은 어떤(주로 시간이나 시행) 변수를 입력받아 관측된 확률변수의 값을 출력하는 함수와 같다. 이제 연속적 확률과정과 이산적 확률과정을 수학적으로 형식화하면 아래와 같다.

- 확률변수 $X$
- 이산적 확률과정 $\{X_n \mid n=0,1,2, \cdots\}$
	- 확률변수의 수열과 같다
- 연속적 확률과정 $\{X_t \mid t \in[0, \infty)\}$


> 함수와 수열
> 수열은 정의역이 자연수인 함수다.


## 예시 코드: 브라운 운동
아래는 가장 흔하게 사용되는 확률과정의 예시인 브라운 운동을 표현하는 파이썬 코드다.

```Python
class Brownian():  
	def __init__(self,x0=0):  
		assert (type(x0)==float or type(x0)==int or x0 is None), 'error'
		self.x0 = float(x0)  
  
	def gen_random_walk(self,n_step=100):  
		w = np.ones(n_step)*self.x0
		for i in range(1,n_step):   
			yi = np.random.choice([1,-1])  
			w[i] = w[i-1]+(yi/np.sqrt(n_step))  
		return w  
  
	def gen_normal(self,n_step=100):  
		# 결과물이 표준분포를 따르도록 유도
		w = np.ones(n_step)*self.x0  
		for i in range(1,n_step):  
			yi = np.random.normal()  
			w[i] = w[i-1]+(yi/np.sqrt(n_step))  
		return w
```


# References
- https://freshrimpsushi.github.io/posts/stochastic-process/
- https://towardsdatascience.com/brownian-motion-with-python-9083ebc46ff0