# Intro
#TODO 

# 1. 헤비안 학습 Hebbian learning
"Cells that fire together, wire together"
$$\Delta w_{i} = \eta x_{i} y$$
- $\Delta w_{i}$ 뉴런 $i$가 가진 synaptic weight의 차이
- $x_{i}$ 입력 신호
- $y$ 출력 신호; Postsynaptic response
- $\eta$ Learning rate


## Drawback과 Oja's rule
동시에 흥분하는 뉴런들 사이의 synaptic weight $w$는 항상 증가하기만 한다. 다음의 Oja가 만든 식을 보면 normalization term이 추가되어 $w$의 무한한 순증가를 막는 것을 알 수 있다.
$$
\begin{aligned}
&w_i^{t+1}=w_i^t+\eta x_i y \\
&w_i^{t+1}=\frac{w_i^t+\eta x_i y}{\left(\sum_{i=1}^n\left(w_i^t+\eta x_i y\right)^2\right)^{\frac{1}{2}}}
\end{aligned}
$$


### Oja's rule과 weight
Oja가 만든 normalized weight에 몇 가지 가정(#TODO)을 추가하고 식을 정리하면, weight $w$는 input $X$의 covariance matrix $C$가 된다. 곧, $w_{i}$는 $C$의 Eigenvector다. Oja의 이론에 따르면 신경망의 가중치 $w$는 input $X$의 PCA 와 같다.
$$
\frac{d}{d(t)} w_i^t=C w_i^t-\left(\left(w_i^t\right)^T C w_i^t\right) w_i^t
$$


## 신경과학 관점: Spike Time Dependent Plasticity
![[assets/Pasted image 20220907205519.png]]
Presynaptic neuron이 흥분하고, 그에 이어서 postsynaptic neuron이 흥분하면 가중치가 증가하며, 이때 시간 각견이 짧을수록 더 가중치를 크게 증가시킨다. 반대로, postsynaptic neuron이 먼저 흥분하고 그에 이어서 presynaptic neuron이 흥분한다면 가중치를 깎는다.


# 2. 역전파 Back propagation
#TODO 


# References
- Donald Hebb, 1949