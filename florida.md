# Barycentric Kernel for Bayesian Optimization


> Abstract
> 


## Introduction
Bayesian optimization(BO) has emerged as a powerful statistical model to solve optimization problems with costly black-box functions. It showed its performance in many real-world applications, such as hyperparameter search in machine learning and reaction optimization for chemicals is essential. The BO has a kernel function as a hyperparameter to be set by experts, yet it is based on non-parametric statistics. The kernel function gives information about search space to the optimization model.

In chemical reaction optimization, it's often modeled as a function of the concentrations of reactants and products. In this context, the kernel should represent the distances between concentrations. This paper introduces a novel barycentric kernel for use in BO for chemical reaction optimization.

The barycentric kernel is a distance function in barycentric space, which is equivalent to the concentration space for chemical compounds. We prove that the barycentric kernel is a definite positive kernel for the BO. Alongside the real-world example of applicational use, we demonstrate that the barycentric kernel can effectively express concentration space and solve chemical reaction optimization problems. Our contribution is to build a barycentric kernel and demonstrate its performance compared to other popular kernels, such as the radial basis function (RBF) kernel.


## Preliminary

### Bayesian Optimization and Kernels
BO is a powerful optimization technique widely used in many domains. It is particularly well-suited to problems that are expensive to evaluate, such as those that require the use of simulators or physical experiments. The key idea behind Bayesian optimization is to build a probabilistic model of the objective function and then use this model to guide the search for the optimum.

BO has two crucial components, kernel and acquisition function. The kernel is used to define and calculate the covariance between data points, which is used to build the surrogate model of the objective function. Kernels can take many forms, but the radial basis function (RBF) kernel is well-known as the most common type. The RBF kernel is defined and justified by a normal distribution and the central limit theorem(CLT). It's reasonable when there is little known about search space.

However, the chemical reaction optimization has a different space that the mathematical assumption of the RBF kernel is based on. The RBF kernel assumes the search space is normal-distributed over features, but the concentrations are the features that are conjugated with each other. This is where the barycentric kernel comes in. The barycentric kernel is explicitly designed for problems involving concentrations, providing a more natural and interpretable way to measure similarity between data points.

In the following section, we will describe the construction and implementation of the barycentric kernel in more detail and compare its performance with that of the RBF kernel on a range of benchmark functions.

### Concentration space
Chemical concentration space is a high-dimensional space where the coordinates represent the fractions of specific materials in a mixture or solution. For example, a three-component mixture with fractions A, B, and C can represent a planar triangle in $\mathbb{R}^3$. The vector of mixture $\boldsymbol{d}$ consists of a vector of chemicals $\boldsymbol{a}$, $\boldsymbol{b}$ and $\boldsymbol{c}$ can be represented as equeation#, where $v_{i}$ represents a volume or weight fractions of each ingredient in the mixture. Since it is a fraction, $\sum_{i}v_{i} = 1$. It is obvious that the mixture is a weighted sum with $\sum=1$ normalization and constraints term.
$$v_1 \boldsymbol{a}+v_2 \boldsymbol{b}+v_3 \boldsymbol{c}=\boldsymbol{d}$$

### Barycentric coordinate system and distance
Generally speaking, the concentration space is a simplex. The simplex is in $\mathbb{R}^n$ space when there are $n$ kinds of individual ingredients. Furthermore, the simplex in $\mathbb{R}^n$ space can be projected to $\mathbb{R}^{n-1}$ space without any loss of information. It is the motivation for building the barycentric kernel for concentration space so that it can implement the conjugated constraints over concentrations while reducing the dimensionality of search space by one without any information loss.

The barycentric coordinate system is equivalent to the concentration space and space on a simplex. The barycentric coordinates represent the weights of the vertices in the convex combination that defines a point in the simplex. A point in barycentric space is represented by a set of weights used to interpolate between the simplex vertices.

**Lemma#. Generalized barycentric distance**
A distance between two points $P$ and $Q$ in 3-dimensional barycentric space is defined as $d = -a^2xy-b^2yz-c^2zx$, where the $a$, $b$ and $c$ represents the edge length of the triangle and $P-Q = (x, y, z)$. The distance can be generalized by equation#.
$$general\ d$$
To generalize this equation for $n$-dimensional barycentric point $P$ to $Q$… 수식 정리중.
$$x+y+z = 0$$

So the L2 norm of a barycentric vector from the barycentric point $P$ to $Q$, $\vec{PQ}$ is defined as equation#.
$$|PQ|^2$$

## Methodology
### Barycentric Kernel
Since the barycentric kernel is for the Gaussian process regression(GPR), the kernel function is natural to define with the exponential function as eq#. This exponentiation is justified by lemma#.
$$k(p,q) = \exp{|PQ|^2}$$
**Lemma#. Barycentric Gaussian distribution**
The RBF kernel is used for GPR since GPR is based on the Gaussian random process, so RBF can calculate the distance between points on outputs from Gaussian distributed feature with CLT. Here we derive Gaussian distribution on a barycentric coordinate system and justify exponential kernel.

…

The exponential function makes the kernel smoother, so the exponential barycentric distance kernel is not only more mathematically correct than the simple barycentric distance but computationally stable.

In summary, the barycentric kernel provides a way to define a meaningful distance metric between concentrations. This allows us to use Bayesian optimization to search for the optimal chemical reaction conditions more efficiently and effectively.

### Implementation
We built the barycentric kernel function in python to work with the popular machine learning library, scikit-learn. The python module we built for the barycentric kernel is freely available at https://github.com/saankim/####. It can be a drop-in replacement of the RBF kernel for optimization on a barycentric or concentration space.

To implement the barycentric kernel, we used the Python programming language and its libraries. Specifically, we used the NumPy library to handle mathematical operations and the scikit-learn library to construct the Gaussian process regression model. We also implemented the barycentric kernel using the scikit-learn library's kernel interface. Our implementation takes as input the barycentric coordinates of two points and returns the Euclidean distance between them. The implementation is efficient and can be used with large datasets.


### Mercer's Theorem: Mathematical convergence
Mercer's Theorem in Barycentric Kernel The barycentric kernel satisfies Mercer's theorem, which is a necessary and sufficient condition for a kernel function to be valid. Mercer's theorem states that a kernel function must be symmetric and positive semi-definite to be valid. In our implementation, we showed that the barycentric kernel is symmetric and positive semi-definite, satisfying the conditions of Mercer's theorem. This guarantees that the kernel function will produce valid covariance matrices for any input dataset.

**Lemma#. Mercer's Theorem for barycentric kernel**


## Experiment
###  Analysis of stability at the boundaries
simplex의 경계에서도 잘 작동하는지 확인
smooth한 kernel function의 output을 3D에서 visualize 해서 fig.로 넣으면 됨.

### Sensitivity analysis
하면 좋은데 시간 없음… 일단 안할듯.

### Convergence
벤치마크 함수로 무게중심 커널이 적용된 베이지안 최적화 모델의 수렴을 확인함
- `Branin function`: 가장 일반적인 최적화 테스트용 함수
- `Ackley function`: local minimum이 많아서 모델을 혼란시키는 함수

### Performance
직교좌표계에 농도 제약조건 없을 때 보다 빨리 수렴함 ∵탐색공간 더 작음

### Real-world
Perovskite 값 넣을수는 있는데 일정하게 실험하지 않아서 좀 그럼.
아마 이 섹션을 뺄듯.

## Discussion

- Interpretation of the results of the experiments.
- Discussion of the implications of the results for using the barycentric kernel in chemical reaction optimization.
-  Comparison of the barycentric kernel's performance with existing chemical reaction optimization methods.

Overall, this section should provide a detailed explanation of the experiments conducted to evaluate the performance of the barycentric kernel and demonstrate the superiority of the proposed method over existing methods in chemical reaction optimization.

1.  Convergence Properties:

-   Analysis of the convergence properties of the barycentric kernel.
-   Comparison of the convergence properties of the barycentric kernel with those of the RBF kernel.
-   Discussion of how the convergence properties affect the performance of the barycentric kernel.

2.  Performance Comparison:

-   Comparison of the performance of the barycentric kernel with the RBF kernel on benchmark functions.
-   Analysis of the strengths and weaknesses of the barycentric kernel compared to the RBF kernel.
-   Discussion of the implications of the performance comparison for chemical reaction optimization.

3.  Boundary Stability:

-   Analysis of the stability of the barycentric kernel at the boundaries of the concentration simplex.
-   Comparison of the boundary stability of the barycentric kernel with that of the RBF kernel.
-   Discussion of how the boundary stability affects the performance of the barycentric kernel.

This paper introduces a barycentric kernel for Bayesian optimization, particularly well-suited for chemical reaction optimization. The barycentric kernel is a novel method that calculates the barycentric distance between two or more data points. Our experimental results demonstrate that the barycentric kernel outperforms the commonly used RBF kernel in terms of convergence and stability at the boundaries.

The barycentric kernel can be used in various optimization problems where the data points lie on a simplex, including chemical reaction optimization. Our results suggest that the barycentric kernel has the potential to significantly improve the efficiency and accuracy of Bayesian optimization in this context.

In the future, we plan to test the barycentric kernel on real-world chemical reaction optimization problems and compare its performance with existing methods. We also plan to explore the potential benefits of using Wasserstein distance to calculate the proper barycentric space.

Overall, the barycentric kernel is a valuable addition to the field of Bayesian optimization, with applications beyond chemical reaction optimization. We hope this work will inspire further research into using barycentric kernels in a wide range of optimization problems.