# Supervised, Unsupervised, Self-supervised
지도학습, 비지도학습, 자가지도학습의 차이는 무엇일까?

인공지능 분야를 처음 공부하면, 다음과 같은 정의에 익숙할 것이다.
- 지도학습: 분명한 라벨과 에러(또는 리워드)를 따라서 학습
- 비지도학습: 분명한 라벨과 에러 없이 학습
- 자가지도학습: 지도학습과 비지도학습의 중간 정도의 방법.

이러한 정의는 아주 당연한 의문을 불러일으킨다. 일반적인 순전파신경망(feed-forward neural network)에서 [[../../idea/learning-algorithms/learning-algorithms|역전파]] 알고리즘은 에러를 weight로 편미분한 값을 출력층부터 입력층까지 전달해 weight 업데이트, 곧 학습을 진행한다. 그런데 label이 없다면 무엇을 기준으로 error를 만들고 BP를 진행해서 weight를 업데이트하고 학습을 달성할수 있다는걸까?


# 지도는 어디에서 일어나는가
지도학습의 지도는 출력된 label과 실제 label의 차이에서 시작된다. 이는 다시 말해서, label이 데이터와 함께 주어졌다는 것이다. 여기에 맞춰서 정의를 좀 더 업데이트해 보자.
- 지도학습: 입력과 올바른 출력의 쌍, 다르게 말해서 데이터와 label의 조합이 주어진 상태로 학습.
- 비지도학습: 입력에서 올바른 label이 주어지지 않았음
- 자가지도학습: label을 스스로 생성함

실제로 여러 자가지도학습 알고리즘을 살펴보면 어떤 형태로든 입력값과 중간층의 출력에 대한 label을 형성하는 것을 볼 수 있다. 주어진 데이터셋을 통해서 지도학습과 비지도/자가지도 학습을 구분하는 이 접근은 조금 더 타당해보인다.

결국 지도학습 기계가 학습하고자 하는 것은 결국 data가 주어졌을때 label의 확률분포다.
$$
\text{supervised-model} \sim  P(label \mid data)
$$


# Unsupervised vs. Self-supervised
비지도 vs. 자가지도

Yann Lecun과 다른 AI연구자들이 언급하거나 본인들의 논문에서 묘사하는 것을 보면, 어느날부터 비지도학습을 자가지도학습이라고 고쳐 부르고 있는 모습을 볼 수 있다. 이제 혼동을 피하기 위해서 비지도학습이 아니라 자가지도학습이라고 부르겠다.


## What makes label in self-supervised learning?
자가지도학습에서 라벨은 어디서 기인하는가?

자가지도학습은 입력된 data의 representation을 어떤 식으로든 생성한다. 이 representation을 일종의 label로 볼 수 있다. 자가지도학습도 지도학습과 비슷하게, 데이터로부터 라벨을 형성한다 $\sim P(label \mid data)$. 다만 라벨이 dataset에t 주어지지 않기 때문에 라벨을 스스로 생성해야한다. 대부분의 현대적 자가지도학습 모델은 라벨의 분포가 data의 분포와 유사하도록 학습한다.
$$
\text{self-supervised-model} \sim  P(label \mid data), P(label) \sim P(data)
$$

But how??

자연스럽게, $\text{self-supervised-model}: data \rightarrow label$ 의 함수로 사용할 수 있기 때문에 self-supervised learning을 pre-train 과정에서 사용할 수 있다.


# Discussion
self-supervised learning은 미묘하게 시간이나 정보의 위계에 대한 묘사를 포함하고 있는 것으로 보인다. 최초의 train/inference 과정에서도 error/reward를 얻어낼 수 있는 지도학습 알고리즘과 달리, self-supervised learning 과정에서는 시간에 따라 계속해서 입력되는 데이터를 stochastic하게 inference해서, higher hierarchy에 있는 data structure를 모델에 학습시키는 과정이기도 하기 때문이다.


자가지도학습이 데이터의 분포를 학습하기 때문에, 데이터 분포에 대한 self-supervised-error의 공간을 상상할 수 있다. 이 공간에서 error를 energy로 보면, self-supervised learning 과정을 [[Energy based models|energy optimization 과정]]으로도 볼 수 있다.


# References
- https://www.reddit.com/r/MachineLearning/comments/q0cex6/d_help_me_understand_selfsupervised_learning/
- https://www.facebook.com/yann.lecun/posts/10155934004262143