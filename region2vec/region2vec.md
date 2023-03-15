# Region2Vec
Community Detection on Spatial Networks Using Graph Embedding with Node Attributes and Spatial Interactions

- [Github](https://github.com/GeoDS/Region2vec)
- [Journal](https://research.com/conference/sigspatial-2021-acm-sigspatial-international-conference-on-advances-in-geographic-information-systems)
	- IF 4.62
	- H5-index 25


## 1. Purpose
GCN으로 geometric dataset에서 community detection 문제를 풀자.


### 1.1. GNN
- Message passing
	- Aggregation
	- Update

$$
h_u^{(k+1)}=\operatorname{UPDATE}^{(k)}\left(h_u^{(k)}, A G G R E G A T E^{(k)}\left(\left\{h_v^{(k)}, \forall\ v \in \mathcal{N}(u)\right\}\right)\right)
$$

- GCN
	- Convolutinoal layer
	- Node $\mathcal{N}$ ⇒ vector $\mathbb{R}^n$projection
		- Supervised learning with labeled dataset
		- Learning node feature embedding


### 1.2. Community detection
- Clustering
	- Unsupervised learning
	- Optimizing attribute-similarity and spatial interaction trade-off


## 2. Methodology
### 2.1. Notation
- $G=(V, E)$
- $\boldsymbol{A}=\left[a_{i j}\right]_{n \times n}$
- $S=\left[s_{i j}\right]_{n \times n}$
	- spatial interaction matrix
	- $s_{ij}$: flow intensity between $v_{i}$ and $v_{j}$
- $K$ commmnuities with label $c_{1\dots K}$


### 2.2. Dataset
- SafeGraph business venue database
	- 일종의 상권 데이터베이스
		- Point of interest(상권)을 node
		- 사이의 통행로를 edge
		- 통행량과 방향을 제공하는 directed graph dataset
- TIGER/Line Shapefiles
	- Region의 경계
	- Rook-type contiguity relationship: sharing borders → edges
- U.S. Census American Community Survey
	- poverty population
	- race/ethnicity
	- household income


### 2.3. Algorithm
클러스터링 목표
1. 비슷한 속성을 공유할 것
2. 지역들이 긴밀히 연계되어 있을 것
3. 지역들이 서로 이어져 있을 것


#### 2.3.1. Stage One: GCN
2-layer GCN에서 각 레이어의 forward propagation은 아래와 같이 쓸 수 있다.
$$
Z^{(1)}=\operatorname{ReLU}\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X W_0\right) ; Z^{(2)}=\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} Z^{(1)} W_1
$$
- Degree matrix인 $D$는 node feature에 대한 일종의 normalization term으로 사용된다.

여기서, GCN은 self-supervised model이지 community detection을 위한 모델이 아니기 때문에  아래 요소들을 조합한 가중치를 error로 제시해줘야 community detection을 위한 학습을 일으킬 수 있다.
- spatial interaction
- flow strength
각각의 interaction이 강할수록 positive pair로 취급. latent space에서 embedding을 가까워지게 했다.

추가로, spatial contiguity를 확보하기 위해 negative node pair가 지리적으로 가까울 때 거리에 비례하는 threshold를 embedding vector의 거리에 준다.
- geographic distance

$$
L_{\text {hops }}=\sum \frac{\mathbb{I}\left(\text { hop }_{i j}>\epsilon\right) d_{i j}}{\log \left(\text { hop }_{i j}\right)} ; \text { Loss }=\frac{\sum_{p=1}^{N_{\text {pos }}} \log \left(s_p\right) d_{\text {pos }}^p / N_{\text {pos }}}{\sum_{q=1}^{N_{\text {neg }}} d_{\text {neg }}^q / N_{\text {neg }}+L_{\text {hops }}}
$$
결과로 위와 같은 Loss function $L$을 제시한다.
- $\text{hop}_{ij}$는 두 노드를 있는 최단 경로의 갯수
- $d_{ij}$는 node embedding representation 사이의 유클리디언 거리
- $\mathbb{I}$ 는 인접성이 없으면 $L_{\text{hops}}$가 정의되지 않도록 하기 위한 `if` term
- $s_{p}$ 값의 range가 커서 $\log$스케일링
- $\text{Loss}$는 아래 조건에서 커진다
	- positve pair의 거리가 멀수록
		- 갯수로 normalized
		- $s_{p}$로 가중치
	- negative pair의 거리가 가까울수록
	- negative pair의 graph hopping이 적을수록


> [!NOTE] Contribution
> - Representation learning을 위한 모델을 clustering에 쓸 수 있게 loss를 수정
> - adj. matrix, node feature 뿐만 아니라 heterogeneous edge data를 잘 녹여냄


#### 2.3.2. Stage Two: Clustering
Bottom-up 방식으로 가장 인접한 cluster로 merge 해 들어간다. 이때 merge 기준은
- embedded feature vector의 거리 
- spatial 인접 여부

![[Algorithm.png]]

![[workflow.png]]


## 4. Baseline
- Louvain algorithm
	- 휴리스틱 modularity optimization.
	- Flow connections 데이터만 사용
- DeepWalk: Random walk based model 1
- Node2Vec: Random walk based model 2
	- GCN 대신 사용. clustering stage는 동일하게 진행
	- spatial adj. maxtrix만 사용
- LINE: Large-scale Information Network Embedding
	- spatial adj. matrix만 사용
- K-Means
	- graph structrue 없이 학습되지 않은 node feature만 입력해서 사용


> [!NOTE]
> 모델마다 제공받는 데이터가 달라 공정한 비교가 어려움


### 4.1. Evaluation
- Intra/Inter Flow Ratio
	- spatial interaction flow ratio
$$
\text { Intra/Inter Flow Ratio }=\frac{\sum_{c_i=c_j} s_{i j}}{\sum_{c_i \neq c_j} s_{i j}} ; c_i, c_j \in 1,2, \cdots, K
$$
- Inequality
	- infrastructure inequality across multiple geographic regions
		- 1: max inequality
		- 0: min inequality
$$
I=\frac{\sigma}{\sqrt{\mu(1-\mu)}} ; 0<\mu<1
$$
- Similarity Metrics
	- cosine similarity (L2-norm dot product)
- Homogeneity Scores
	- socio-economic characteristics
		- percentage of the population with income at or lower than 200% federal poverty level


## 5. Results
![[result table.png]]
- Intra-inter flow ratio
	- Louvain is the best
	- Region2Vec is second
- Inequality
	- K-Means is the best*
	- Region2Vec is second
- Consine Similarity
	- K-Means is the best*
	- Region2Vec is second
- Homogeneity
	- K-Means is the best*
	- Region2Vec is second

K-Means는 원래 inequality 최소화와 similarity 최대화를 위한 알고리즘 이다. Intra-inter flow ratio를 보면 spatial information을 이용해 적절한 클러스터를 만들지 못했음을 볼 수 있다.

Louvain은 modularity score를 고려해서 greedy 하게 local optimization 하는 알고리즘이기 때문에 Intra-inter flow ratio 수치에서 유리하다.


## 6. Conclusion
- GCN 을 사용해서
	- spatial adj.
	- spatial interaction flow
	- node attributes
- 를 고려한 clustering method를 만들 수 있었음


> [!NOTE] 
> Node embedding의 차원을 어떻게 결정했는지 논문에 나오지 않는다. 차원이 높으면 $d_{ij}$의 변별력이 없어질 수 있어서 중요한 변수로 생각. 기본값은 14.


> [!NOTE] GNN으로 community detection
> - Loss 조정해 GNN 학습/임베딩 후 ML clustering
> - Class의 prob. distribution을 학습시킴
> - Modularity를 직접 예측

