# Qiskit으로 시작하는 양자머신러닝
양자 머신러닝에는 알고리즘/데이터가 각각 양자/고전인지에 따라 네 가지 유형이 있다. 
- 알고리즘/데이터
	- CC
		고전 데이터를 고전 연산으로 처리
		QML의 측면에서는, [Ewin Tang. 2019](https://dl.acm.org/doi/10.1145/3313276.3316310) 와 같이 양자에서 영감을 받은 알고리즘을 사용한다. 
	- QC
		고전 데이터를 양자 알고리즘으로 처리
		QML의 주요 관심 분야
		아래 두 가지 하드웨어에 기반한 알고리즘이 연구되고 있다.
		- qRAM
			양자 메모리를 이용해 중첩된 데이터에 다항시간에 접근하는 기술을 바탕으로 한 알고리즘.
			현재 qRAM을 구현할 수 있는 하드웨어 후보는 없다.
		- Quantum computing
			양자 컴퓨터에 기반한 알고리즘.
			**이 문서에서 주로 집중할 분야**
	- CQ
		양자 데이터를 고전 알고리즘으로 처리
		양자컴퓨팅을 구현하는데 있어 필요하다.
		- Characterisation([Usman, 2020](https://www.nature.com/articles/s41524-020-0282-0))
		- Control([Niu, 2019](https://www.nature.com/articles/s41534-019-0141-3))
		- Discriminating([Magesan, 2015](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.114.200501))
	- QQ
		양자 데이터를 양자 알고리즘으로 처리
		아직 초기 연구 단계


## Parameterized Quantum Circit
Parameterized Quantum circuit을 통해서 QML 알고리즘을 구현하는 회로를 만들 수 있다.


### Notation
- Parameterized Quantum Circit
- Ansatzes
- $\ket{\phi_\theta} = \mathbf{U}_\theta \ket{\phi_0}$
	아래에 위 수식에 대한 설명이 있다.


### 가장 단순한 형태
- 매개 변수 게이트 두 개
- 같은 파라미터를 가진
	- 단일 큐비트 회전 게이트 한 개
	- 제어-회전 게이트 한 개
```python
from qiskit.circuit import QuantumCircuit, Parameter
theta = Parameter('θ')

qc = QuantumCircuit(2)
qc.rz(theta, 0)
qc.crz(theta, 0, 1)
qc.draw()
```
- 위 회로를 두 개의 다른 파라미터를 가지도록 수정
```python
from qiskit.circuit import ParameterVector
theta_list = ParameterVector('θ', length=2)

qc = QuantumCircuit(2)
qc.rz(theta_list[0], 0)
qc.crz(theta_list[1], 0, 1)
qc.draw()
```

양자 회로에 사용되는 모든 양자 게이트는 Unitary 연산자$\mathbf{U}_\theta$다. 곧, parameterized quantum circit은 어떤 초기상태 $\ket{\phi_0}$의 Qbit $n$개에 대한 unitary operation으로 볼 수 있다. 따라서 $\theta$가 tunable parameter의 집합일 때, parameterized quantum circit은 다음과 같은 수식으로 나타낼 수 있다. 덧붙이자면, 보통 초기 상태는 $\ket{0}^{\otimes n}$로 설정된다.
$$\ket{\phi_\theta} = \mathbf{U}_\theta \ket{\phi_0}$$


### Parameterized quantum circuit properties
Machine learning을 위해 사용하기 위해서는 Quantum circuit이 생성하는 output Quantum state의 집합이 데이터의 해석을 위한 의미있는 집합이어야 한다. 이때 양자회로가 Qbit의 Quantum entanglement를 포함하고 있다면 고전 컴퓨터로 모사하기가 어려워진다.
[Sukin Sim, 2019](https://learn.qiskit.org/course/machine-learning/parameterized-quantum-circuits)에서 quantum circit을 expressibility와 entangling capability에 따라 분류하였다.


#### Expressibility

#### Entangling capability

#### Hardware efficiency


## Reference
- ["Quantum Machine Learning," Qiskit, 2022](https://qiskit.org/learn/course/machine-learning-course)
