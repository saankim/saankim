# Wasserstein distance

와셔스테인 거리는 두 분포를 전환하는 최소일의 양을 metric으로 하는 거리함수.

```tikz
\begin{document}
\begin{tikzpicture}[scale=3]
% coordinate
\draw[->] (0,0) -- (1.5,0) node[right] {$x$};
\draw[->] (0,0) -- (0,1.5) node[above] {$y$};
% L1 norm
\draw[thick, red, domain=(0:1),smooth,samples=100] plot (\x,{(1-\x});
% L2 norm
\draw[thick, blue, domain=(0:1),smooth,samples=100] plot (\x,{(1-\x^(2))^(1/2)});
% L3 norm
\draw[thick, teal, domain=(0:1),smooth,samples=100] plot (\x,{(1-\x^(3))^(1/3)});
% L1/2 norm
\draw[thick, orange, domain=(0:1),smooth,samples=100] plot (\x,{(1-\x^(1/2))^(2)});
\end{tikzpicture}
\end{document}
```
