# Modern AI in the Perspective of Computational Neuroscience
Perception: convolution of receptive fields
- AlexNet
- GoogleNet
- VGGNet
- ResNet
- Mask-RCNN
- YOLO

Time: sequence modeling
- RNN
- LSTM
- GRU

Attention: key-query-value
- Seq2Seq
- Transformer
	- GPT2
	- BERT

Generative: latent representation
- VAE
- GAN
- Flow-based

Active-Value: RL
- DQN
- A3C
- World model
- Dreamer

Relation: graph
- CGNN
- RGNN
- GAE
- GNN-RL


Contrasive: prediction in the latent space
- Contrasive Predictive Coding(CPC)


# ResNet
Residual connection(skip connection)으로 gradient descent 문제를 일부 해결해서 깊은 NN을 만들 수 있었다.
![[resnet.png]]

# CNN
Spatial inductive bias를 가지고 있는 convolution filter를 이용해서 image recognition 분야에서 성능을 끌어올릴 수 있었다.


## Mask RCNN
![[rcnn.png]]
CNN을 거쳐서 나올 feature map에 mask를 씌워서 segmentation.


Task의 hierarchy에 따라 분류한 vision 모델의 종류
- Image recognition
- Semantic segmentation
- Object detection
- Instance segmentation

# Self-attention mechanism

Word2Vec

ViT: Vision Transformer

Generative models
variational autoencoder
![[gan.png]]
- 본래의 분포를 따르기: 원래의 오토인코더
- 그런데 각각의 분포가 가우시안 분포하면 좋겠어
	- 이 덕분에 representation이 latent space 위에서 랜덤하게 퍼져있는게 아니라, 가우시안하지만 최소한의 overlap만 가지도록 해서 빽빽하게 채워지게 된다.
	- 이걸 probabilistic modeling이라고 부른다
이러한 접근방식이 헬름홀츠 머신에서 제시된 적 있다.
인코더 디코더의 나비넥타이 구조를 겹쳐둔게 헬름홀츠 머신이기 때문이다.
- 헬름홀츠 머신은 weight를 공유하지만 오토인코더는 인코더-디코더가 weight를 공유하지는 않는다.

GAN은, 비슷한 구조를 가진다.
그런데 input으로 노이즈 Z를 주고, 이거에서 fake x를 뽑는데, 이 fake x를 discriminator가 real fake 구분하기 어렵게 자꾸 학습한다. 
![[vae and gan.png]]


Dreamer2
simulative 한 이해를 가질 수 있도록 학습


GNN
tranductive learning으로 인과관계를 인코딩 가능


Contrasive learning
![[contrastive learning.png]]


총정리
![[landscaping.png]]


missing
1) Epistemic and Pragmatic value를 어떻게 정의하고, 이 두 가지 value processing component를 탑재한 agent를 어떻게 구현할 것인가?
-   -  Epistemic value: 세상으로부터 얻을 수 있는 유의미한 지식 à (pseudo) 인과 관계
-   -  Pragmatic value: Agent의 궁극적인 목표 설정 (energy efficiency?)à외재적, 내재적 가치 형성

    2) Model-free와 Model-based system을 spectrum을 어떤 식으로 구현할 것인가?
-   -  Internal generative model: p(st+1|st) distribution을 어떻게 학습할 것인가? Latent space에서의 prediction을 구현하는 가장 유용한 방법은 무엇일까?
-   -  Habitual system(model-free)은 어떤 기제를 바탕으로 만들 것인가?
-   -  두 가지 큰 predictive representation, reward prediction과 state prediction error를 계산하는 기제는 어떤 식으로 구현할 것인가?
    
    3) 내·외부 환경 변수 representation, 즉 외부 세상의 상태 (external state)를 나타내는 변수와 agent의 내부 상태 (internal state)을 어떤 식으로 개념화해야 할까?
-   -  Attractor를 가지고 있는 state space에서의 oscillation?
-   -  Hierarchical structure를 가지고 있는 state variables들?
    
    4) 평생 학습 (Life-long learning)을 하기 위한 기제를 어떻게 구현할 것인가?
-   -  Latent representation을 어떻게 형성시켜야 각기 다른 학습을 서로 간섭하지 않게 만들 수 있을까?
-   -  Curriculum learning의 가장 좋은 전략은 무엇인가?


# References
- https://www.youtube.com/watch?v=VWIfhuMxBS4