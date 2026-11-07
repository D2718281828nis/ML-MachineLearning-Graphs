# Graph connectivity versus local methods: problem statement and formulas / Связность графа против локальных методов: постановка задачи и формулы {-}

> **What this is / Что это.** The formal model behind
> [`examples/graph_connectivity_chains_circles.ipynb`](../../examples/graph_connectivity_chains_circles.ipynb)
> and the case study [ML-overlay-problem](https://github.com/D2718281828nis/ML-overlay-problem).
> Every formula has a title, its meaning in English and Russian, and a label `{#eq:…}`;
> the notebook uses the same LaTeX with the same numbers. / Формальная модель к
> примеру и кейсу: у каждой формулы есть название, смысл на английском и русском и
> метка; в notebook те же формулы с теми же номерами.
>
> **Format / Формат.** Display formulas are in double dollar signs with the label on the
> closing line, references are `@eq:label`, inline formulas are `$…$`. The single
> first-level heading below numbers the formulas (1.1)…(1.16) when the file is converted
> with `md-to-word-equations`: `python3 md2docx_eq.py README.md -o graph_connectivity.docx`.
>
> **Status / Статус.** Statements 1–4 are elementary facts for the idealised model
> @eq:chains (exact step, no noise, nearly straight chains); the notebook checks each of them
> numerically on the CSV data. They are teaching statements, not published theorems. /
> Утверждения 1–4 – элементарные факты для идеализированной модели @eq:chains; notebook
> проверяет их на данных. Это учебные утверждения, а не опубликованные теоремы.

# Graph connectivity versus local methods

## 1. Problem statement / Постановка задачи {-}

**Data / Данные.** $n$ points in the plane with a hidden binary class; red ($y=1$) is the
minority with share $p$. / $n$ точек на плоскости со скрытым классом; красный класс
($y=1$) – меньшинство с долей $p$.

$$
\mathcal D=\left\{\left(\mathbf x_i,y_i\right)\right\}_{i=1}^{n},\qquad \mathbf x_i\in\mathbb R^{2},\quad y_i\in\{0,1\},\qquad p=\frac1n\sum_{i=1}^{n}y_i.
$$ {#eq:data}

- In the data: $n\approx1\,670$, $p=0{,}136$ (227 red points).

**Task / Задача.** An unsupervised method computes a score $s_i$ from the coordinates only
and flags the $n_{\mathrm{flag}}$ highest scores. The colours $y_i$ are used only for
evaluation; the prior $p$ is assumed known. / Метод без учителя вычисляет оценку $s_i$
только по координатам и отмечает $n_{\mathrm{flag}}$ наибольших; цвета используются
только для оценки качества.

$$
s_i=s\left(\mathbf x_i;\ \mathbf x_1,\dots,\mathbf x_n\right),\qquad \hat y_i=\mathbb 1\left[\operatorname{rank}\left(s_i\right)\leqslant n_{\mathrm{flag}}\right],\qquad n_{\mathrm{flag}}=\operatorname{round}(pn).
$$ {#eq:task}

**Quality / Качество.** Precision, recall and $F_1$ of the red class. / Точность, полнота
и $F_1$ красного класса.

$$
\mathrm P=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\qquad \mathrm R=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},\qquad F_1=\frac{2\,\mathrm P\,\mathrm R}{\mathrm P+\mathrm R}.
$$ {#eq:f1}

**Trivial levels / Тривиальные уровни.** Flagging every point gives $\mathrm P=p$,
$\mathrm R=1$. Flagging $n_{\mathrm{flag}}=pn$ points at random gives
$\mathbb E\,\mathrm{TP}=p\,n_{\mathrm{flag}}$, so $\mathrm P\approx\mathrm R\approx p$. /
Все точки красные: $\mathrm P=p$, $\mathrm R=1$; случайный выбор $pn$ точек: $\mathrm P\approx\mathrm R\approx p$.

$$
F_1^{\text{all red}}=\frac{2p}{1+p}\approx0{,}239,\qquad \mathbb E\,F_1^{\text{random}}\approx p=0{,}136.
$$ {#eq:baseline}

- A method below $F_1^{\text{all red}}$ has learned nothing; a score with many ties (one
  giant component) behaves like random flagging and lands near $p$. / Метод ниже
  $F_1^{\text{all red}}$ ничему не научился; оценка с массовыми совпадениями ведёт себя как
  случайный выбор и даёт около $p$.

**Overlay / Наложение.** The class means and spreads are almost equal in each coordinate
$d$, so methods that use position (K-Means, GMM, a linear boundary) have no signal. /
Средние и разбросы классов почти совпадают, поэтому методы, использующие положение
точек, не имеют сигнала.

$$
\frac{\left|\bar x_d^{(1)}-\bar x_d^{(0)}\right|}{s_d}\ll1,\qquad \frac{s_d^{(1)}}{s_d^{(0)}}\approx1,\qquad d\in\{1,2\}.
$$ {#eq:overlay}

- In the data: $0{,}02$–$0{,}05$ (walk chains) and $0{,}09$–$0{,}19$ (circles) for the
  means; spread ratios $0{,}72$–$0{,}90$.

## 2. The chain model / Модель цепочек {-}

**Chains / Цепочки.** Every point belongs to exactly one chain $\mathcal C_m$ of length
$L_m$ with a constant step $\delta$; different chains are at least $g>\delta$ apart; the
colour is constant on a chain and is determined by its length. / Каждая точка принадлежит
одной цепочке $\mathcal C_m$ длины $L_m$ с шагом $\delta$; разные цепочки отстоят не менее
чем на $g>\delta$; цвет постоянен на цепочке и определяется её длиной.

$$
\mathcal C_m=\left(\mathbf c_{m,1},\dots,\mathbf c_{m,L_m}\right),\quad \left\lVert\mathbf c_{m,t+1}-\mathbf c_{m,t}\right\rVert=\delta,\quad \min_{m\neq m'}\ \min_{t,t'}\left\lVert\mathbf c_{m,t}-\mathbf c_{m',t'}\right\rVert\geqslant g>\delta,\quad y\big|_{\mathcal C_m}=\mathbb 1\left[L_m\geqslant L^{*}\right].
$$ {#eq:chains}

- In the data: $\delta\approx12$ px, $g\approx30$ px; green $L_m\in[9,20]$ (mean 14,6), red
  $L_m\in\{37,38\}$; walk chains are open, circles are closed loops. / В данных:
  $\delta\approx12$ px, $g\approx30$ px; зелёные $L_m\in[9,20]$, красные $L_m\in\{37,38\}$.

## 3. Local methods / Локальные методы {-}

**Local scores / Локальные оценки.** Distance to the $k$-th neighbour, k-NN density and the
local outlier factor depend only on the neighbourhood $N_k(i)$ of the point. / Расстояние до
$k$-го соседа, плотность k-NN и LOF зависят только от окрестности $N_k(i)$.

$$
d_k(\mathbf x_i)=\left\lVert\mathbf x_i-\mathbf x_{(k)}(i)\right\rVert,\qquad \hat f_k(\mathbf x_i)\propto\frac{1}{d_k(\mathbf x_i)},\qquad \mathrm{LOF}_k(i)=\frac{1}{\left|N_k(i)\right|}\sum_{j\in N_k(i)}\frac{\mathrm{lrd}_k(j)}{\mathrm{lrd}_k(i)}.
$$ {#eq:local}

**Statement 1: local indistinguishability / Утверждение 1: локальная неразличимость.**
Under @eq:chains, a point at least $r/\delta$ steps from both chain ends sees, inside any
radius $r<g$, a chain segment of $2\lfloor r/\delta\rfloor+1$ points whatever its colour.
The colour can therefore enter a local score only through chain ends, whose share in class
$y$ is $2M_y\big/\sum_{m:\,y_m=y}L_m$ ($M_y$ chains of that class). / Точка, удалённая от
концов цепочки не менее чем на $r/\delta$ шагов, видит в радиусе $r<g$ отрезок из
$2\lfloor r/\delta\rfloor+1$ точек независимо от цвета; цвет проявляется локально только
через концы цепочек.

$$
\left|B_r(\mathbf x_i)\cap\mathbf X\right|=2\left\lfloor\frac{r}{\delta}\right\rfloor+1\quad(r<g,\ \text{interior }i),\qquad \pi_{\mathrm{end}}^{(y)}=\frac{2M_y}{\sum_{m:\,y_m=y}L_m}.
$$ {#eq:indist}

- In the data: $\pi_{\mathrm{end}}^{(0)}=0{,}137$, $\pi_{\mathrm{end}}^{(1)}=0{,}053$ for the walk
  chains (exactly the observed 13,7% and 5,3%); closed circles have no ends, so
  $\pi_{\mathrm{end}}=0$. This weak hint is why every local method on the walk chains stays at
  or below @eq:baseline. / Этот слабый сигнал – причина, по которой локальные методы на
  цепочках не превышают @eq:baseline.

## 4. The k-NN graph / Граф k ближайших соседей {-}

**Symmetric k-NN graph / Симметричный граф k-NN.** Built from the coordinates only; the
labels never enter $A$. / Строится только по координатам; метки в $A$ не входят.

$$
A_{ij}=\mathbb 1\left[j\in N_k(i)\ \vee\ i\in N_k(j)\right],\qquad i\neq j.
$$ {#eq:knngraph}

**Edge homophily / Гомофилия рёбер.** The share of edges that join the same colour, and the
share of a red node's edges that lead to red; without structure $h_1\approx p$. / Доля
рёбер между точками одного цвета и доля рёбер красной вершины, ведущих к красным; без
структуры $h_1\approx p$.

$$
h(A)=\frac{\sum_{i,j}A_{ij}\,\mathbb 1\left[y_i=y_j\right]}{\sum_{i,j}A_{ij}},\qquad h_1(A)=\frac{\sum_{i,j}A_{ij}\,y_iy_j}{\sum_{i,j}A_{ij}\,y_i}.
$$ {#eq:homophily}

- In the data at $k=2$: $h_1=0{,}33$ (original), $0{,}69$ (ring), $1{,}00$ (chains and circles).

**Statement 2: when the components are the chains / Утверждение 2: когда компоненты
совпадают с цепочками.** Each point must reach both chain neighbours ($k\geqslant2$), and its
$k$ nearest neighbours must all lie on its own chain. The $k$-th own-chain neighbour is
farthest at an open end ($\approx k\delta$) and closest on a closed loop
($\approx\lceil k/2\rceil\delta$). / Каждая точка должна связаться с обоими соседями по
цепочке ($k\geqslant2$), а все её $k$ ближайших соседей должны лежать на своей цепочке.

$$
2\leqslant k,\qquad d_k^{\mathrm{own}}(i)<g\ \ \forall i,\qquad d_k^{\mathrm{own}}\approx\begin{cases}k\,\delta, & \text{open chain (end point)},\\ \lceil k/2\rceil\,\delta, & \text{closed loop.}\end{cases}
$$ {#eq:kcond}

- With $g/\delta\approx2{,}5$: open chains allow only $k=2$; closed circles allow
  $k\in\{2,3,4\}$ ($\lceil 4/2\rceil\cdot12=24<30$, $\lceil5/2\rceil\cdot12=36>30$). This is
  exactly the observed $k$-sweep: walk chains collapse at $k=3$, circles at $k=5$. / Это
  в точности наблюдаемая зависимость от $k$.

**Score: component size / Оценка: размер компоненты.** $\mathcal K(i)$ is the connected
component of $A$ containing $i$. Under @eq:kcond the components are the chains, so the
score is the chain length, and size separation gives a perfect classifier. / При
выполнении @eq:kcond компоненты совпадают с цепочками, и оценка равна длине цепочки.

$$
s_i=\left|\mathcal K(i)\right|=L_{m(i)},\qquad \max_{m:\,y_m=0}L_m<\min_{m:\,y_m=1}L_m\ \Longrightarrow\ F_1=1.
$$ {#eq:compsize}

## 5. Why the signal is long-range / Почему сигнал дальний {-}

**Statement 3: h-hop reach / Утверждение 3: охват за h шагов.** $R_h(i)$ is the number of
points within $h$ hops along the graph; $a_i,b_i$ are the numbers of chain points on the two
sides of $i$. For $h\leqslant\min(a_i,b_i)$ the reach is $2h+1$ for both colours; only beyond
the end of the shorter (green) chains does it differ, and its limit is the component size
@eq:compsize. / Пока $h\leqslant\min(a_i,b_i)$, охват равен $2h+1$ для обоих цветов;
различие появляется только за концом короткой цепочки, а предел – размер компоненты.

$$
R_h(i)=\left|\left\{j:\ d_G(i,j)\leqslant h\right\}\right|=\min(h,a_i)+\min(h,b_i)+1,\qquad \lim_{h\to\infty}R_h(i)=\left|\mathcal K(i)\right|.
$$ {#eq:reach}

- For a point in the middle of a red chain the reach keeps growing up to $h\approx19$ (half
  of 38), which is where the notebook's $F_1(h)$ curve reaches 1. / Для середины красной
  цепочки охват растёт до $h\approx19$, где кривая $F_1(h)$ достигает 1.

**Statement 4: smoothing sees only K hops / Утверждение 4: сглаживание видит только K
шагов.** A GCN-style smoothing of a local score $\mathbf s^{(0)}$ with $K$ steps uses only the
$K$-hop neighbourhood, and in a chain interior a locally constant score is a fixed point
($\tilde d_i=3$). Smoothing therefore cannot recover a signal that needs $h\approx19$
hops. / Сглаживание за $K$ шагов использует только $K$-окрестность, а в середине цепочки
постоянная оценка не меняется; дальний сигнал сглаживание не восстанавливает.

$$
\hat{\mathbf A}=\tilde{\mathbf D}^{-1/2}\left(\mathbf A+\mathbf I\right)\tilde{\mathbf D}^{-1/2},\qquad \mathbf s^{(K)}=\hat{\mathbf A}^{K}\mathbf s^{(0)},\qquad s^{(K)}_i=F\left(\left\{s^{(0)}_j:\ d_G(i,j)\leqslant K\right\}\right).
$$ {#eq:smooth}

## 6. The rule / Правило {-}

**When a graph beats local methods / Когда граф лучше локальных методов.** Two conditions
together: (a) the edges are homophilous, and (b) the class is independent of every small
neighbourhood but determined by a non-local structure such as the component. If (a) fails
(original, ring), propagation mixes the classes; if (b) fails, a local method is just as
good. / Нужны оба условия: (а) гомофильные рёбра и (б) класс не зависит от малых
окрестностей, но определяется нелокальной структурой. Без (а) распространение смешивает
классы, без (б) локальный метод не хуже.

$$
\text{(a)}\ \ h(A)\approx1,\qquad \text{(b)}\ \ \Pr\left(y_i=1\mid B_r(\mathbf x_i)\cap\mathbf X\right)\approx p\ \ (r<g),\qquad \Pr\left(y_i=1\mid\mathcal K(i)\right)\in\{0,1\}.
$$ {#eq:rule}

**Label leakage / Утечка метки.** Edges must be a function of the features only. Edges built
from a label proxy such as `Type` make (a) hold trivially and the result meaningless. /
Рёбра должны зависеть только от признаков; рёбра по `Type` выполняют (а) тривиально и
обесценивают результат.

$$
A=A\left(\mathbf x_1,\dots,\mathbf x_n\right),\qquad A_{ij}\ \text{must not depend on}\ y\ \text{or on any proxy of}\ y.
$$ {#eq:noleak}

# Formula map / Карта формул {-}

| Formula | Meaning / Смысл | Notebook section |
|---|---|---|
| @eq:data, @eq:task, @eq:f1 | problem statement / постановка | 1, 4 |
| @eq:baseline | trivial levels / тривиальные уровни | 1, 4, 7 |
| @eq:overlay | classes overlay / наложение классов | 1, 3 |
| @eq:chains | chain model / модель цепочек | 1 |
| @eq:local, @eq:indist | local methods and their limit / локальные методы | 5, 7 |
| @eq:knngraph, @eq:homophily | graph and homophily / граф и гомофилия | 6 |
| @eq:kcond, @eq:compsize | components = chains / компоненты = цепочки | 8, 10 |
| @eq:reach, @eq:smooth | long-range signal / дальний сигнал | 9 |
| @eq:rule, @eq:noleak | the rule / правило | 12 |
