# 3. Poincaré model and maps / Модель Пуанкаре и отображения


Возьмём небольшое бинарное дерево из семи вершин:

```text
        root
       /    \
      A      B
     / \    / \
   A1  A2  B1  B2
```

Или в виде иерархии:

```text
root -> {A, B} -> {A1, A2, B1, B2}
```

В гиперболическом шаре Пуанкаре такое дерево удобно раскладывать так:

- корень находится ближе к центру;
- вершины уровня `A`, `B` находятся дальше от центра;
- листья `A1`, `A2`, `B1`, `B2` находятся ещё ближе к границе.


> Чем глубже уровень иерархии, тем больше места ему нужно. В гиперболическом пространстве около границы становится всё больше «углового» места, поэтому ветви можно разводить в разные стороны.

- визуально граница диска кажется близкой;
- но гиперболическое расстояние до границы бесконечно.


Шар Пуанкаре кривизны `-c` определяется так:

```math
\mathbb D_c^n
=
\left\{
x \in \mathbb R^n :
c\lVert x\rVert^2 < 1
\right\}.
```

Это значит, что допустимые точки находятся внутри шара радиуса:

```math
\frac{1}{\sqrt c}.
```

То есть если `c = 1`, радиус равен `1`.

Если `c = 4`, радиус равен:

```math
\frac{1}{2}.
```

Если `c = 0.25`, радиус равен:

```math
2.
```

Сама граница:

```text
c||x||^2 = 1
```

в многообразие не входит. Это бесконечно удалённая область.


## Расстояние между двумя точками

Для точек `u`, `v` внутри шара расстояние равно:

```math
d_c(u,v)
=
\frac{1}{\sqrt c}
\mathrm{arcosh}
\left(
1+
\frac{
2c\lVert u-v\rVert^2
}{
(1-c\lVert u\rVert^2)
(1-c\lVert v\rVert^2)
}
\right).
```

Здесь важно:

- `u` и `v` должны быть строго внутри шара;
- знаменатели положительны только если:

```math
c\lVert u\rVert^2 < 1,
\qquad
c\lVert v\rVert^2 < 1.
```

# 3. Сложение Мёбиуса и геодезические координаты 

В обычном евклидовом пространстве мы можем просто складывать векторы:

```text
x + y
```

Но в гиперболическом шаре обычное сложение не сохраняет нужную геометрию. Поэтому используют специальные операции.

---

## Сложение Мёбиуса

Для внутренних точек `u`, `v` определяется операция:

```math
u \oplus_c v
=
\frac{
(1+2c\langle u,v\rangle+c\lVert v\rVert^2)u
+
(1-c\lVert u\rVert^2)v
}{
1+2c\langle u,v\rangle+c^2\lVert u\rVert^2\lVert v\rVert^2
}.
```

Это не обычное сложение векторов. Это специальная операция, согласованная с геометрией шара Пуанкаре.

## Скалярное умножение Мёбиуса

Для точки `x` и вещественного числа `t` определяется:

```math
t \otimes_c x
=
\frac{
\tanh
\left(
t\mathrm{artanh}(\sqrt c\lVert x\rVert)
\right)
}{
\sqrt c\lVert x\rVert
}
x,
```

при:

```math
x \ne 0.
```

Для нуля отдельно полагают:

```math
t \otimes_c 0 = 0.
```

- обычное умножение вектора на число двигает точку вдоль прямой;
- `t ⊗_c x` делает аналогичное движение, но уже вдоль гиперболического луча.

---

## Геодезическая линия между двумя точками

Геодезическая линия — это аналог прямой линии в искривлённом пространстве.

Отрезок геодезической от `u` к `v` можно параметризовать так:

```math
\gamma_{u\to v}(t)
=
u
\oplus_c
\left(
t \otimes_c
\left(
(-u)\oplus_c v
\right)
\right),
```

где:

```math
0 \le t \le 1.
```

При этом:

```math
\gamma(0) = u,
```

```math
\gamma(1) = v.
```

То есть параметр `t` плавно ведёт от начальной точки к конечной.

В модели Пуанкаре геодезические линии обычно выглядят не как прямые отрезки, а как дуги окружностей, которые пересекают границу диска под прямым углом.

Исключение — диаметры диска.

То есть:

```text
прямая линия через центр — геодезическая;
обычная хорда, не проходящая через центр, обычно не геодезическая.
```


### Intuition and tiny graph

Take the seven-node binary tree `root -> {A,B} -> {A1,A2,B1,B2}`. A layout can
place deeper levels at larger Poincaré radii. More angular room becomes available
near the boundary, while metric distance to that boundary diverges. Radius may
therefore correlate with hierarchical depth in a trained embedding, and angle
may separate branches. This is a cautious interpretation, not an invariant: an
isometry can move the apparent center, and a loss function need not encode depth.

### Convention and domain

**Convention box.** Let `n>=1`, `c>0`, and sectional curvature be `K=-c`. Vectors
`x,u,v` use Euclidean coordinates in `R^n`; `||.||` and `<.,.>` are Euclidean.
The Poincaré ball and conformal metric are

```math
\mathbb D_c^n=\{x\in\mathbb R^n:c\lVert x\rVert^2<1\},
\qquad
\lambda_x^c=\frac{2}{1-c\lVert x\rVert^2},
\qquad
g_x^c=(\lambda_x^c)^2 I_n.
```

Thus the coordinate radius is `1/sqrt(c)`. The boundary is not in the manifold.
The distance is

```math
d_c(u,v)=\frac{1}{\sqrt c}\mathrm{arcosh}\!\left(
1+\frac{2c\lVert u-v\rVert^2}
{(1-c\lVert u\rVert^2)(1-c\lVert v\rVert^2)}\right).
```

All denominators are positive only for interior points. At the origin,

```math
d_c(0,x)=\frac{2}{\sqrt c}\mathrm{artanh}(\sqrt c\lVert x\rVert).
```

This follows by substituting `u=0` and using
`arcosh((1+t^2)/(1-t^2))=2 artanh(t)` for `0<=t<1`.

### Möbius addition and geodesics

For interior `u,v`, define

```math
u\oplus_c v=
\frac{(1+2c\langle u,v\rangle+c\lVert v\rVert^2)u
{}+(1-c\lVert u\rVert^2)v}
{1+2c\langle u,v\rangle+c^2\lVert u\rVert^2\lVert v\rVert^2}.
```

For `x!=0` and real `t`, Möbius scalar multiplication is

```math
t\otimes_c x=
\frac{\tanh\!\left(t\mathrm{artanh}(\sqrt c\lVert x\rVert)\right)}
{\sqrt c\lVert x\rVert}x,
\qquad t\otimes_c0=0.
```

A geodesic segment from `u` to `v` can be parameterized by

```math
\gamma_{u\to v}(t)=u\oplus_c\left(t\otimes_c((-u)\oplus_c v)\right),
\qquad 0\le t\le1.
```

Unlike Euclidean chords, these curves generally appear as circular arcs
orthogonal to the boundary circle. Diameters are the special straight case.

### Origin exponential and logarithmic maps

The tangent space at the origin is `T_0 D_c^n ~= R^n`. Under the coordinate
convention above, for nonzero inputs,

```math
\exp_0^c(w)=
\frac{\tanh(\sqrt c\lVert w\rVert)}{\sqrt c\lVert w\rVert}w,
\qquad
\log_0^c(x)=
\frac{\mathrm{artanh}(\sqrt c\lVert x\rVert)}
{\sqrt c\lVert x\rVert}x.
```

The continuous value at zero is zero. The typed maps are
`exp_0^c:T_0 D_c^n -> D_c^n` and `log_0^c:D_c^n -> T_0 D_c^n`.
Substitution gives `log_0^c(exp_0^c(w))=w` in exact arithmetic.

### Limiting case and guardrails

Because `lambda_x^c -> 2` as `c -> 0+`, this convention has

```math
\lim_{c\to0^+}d_c(u,v)=2\lVert u-v\rVert.
```

The factor `2` is not an error. A convention normalized to Euclidean metric
`I_n` would scale distance differently. Numerically:

1. use floating-point inputs and reject `c<=0`;
2. project to radius `(1-eps)/sqrt(c)` after updates;
3. clip `sqrt(c)||x||` below `1` before `artanh`;
4. clamp the `arcosh` argument to at least `1`;
5. test both ordinary and near-boundary points, preferably in `float64`.

**Common misconception.** A chord in a Poincaré plot is not generally a
geodesic. It is a geodesic only on a diameter.

### Repository implementation

| Mathematical object | Repository implementation | State |
|---|---|---|
| `D_c^n`, projection | `inside_ball`, `project` in the curated notebook | Deterministically tested; clean Colab run not yet recorded |
| `d_c` | `poincare_distance` | Identity, symmetry, boundary growth, and Euclidean limit tested |
| `oplus_c`, `otimes_c` | `mobius_add`, `mobius_scalar_mul` | Endpoint/geodesic tests included |
| `exp_0^c`, `log_0^c` | `expmap0`, `logmap0` | Round-trip test included |
| General-point exp/log | Missing example | Planned |
| Lorentz/Klein conversion | Missing example | Planned after source-level convention review |

Run [the curated notebook](../../examples/hyperbolic_graph_basics.ipynb) in
Google Colab. The hyperbolic notebooks in `research-notebooks/` remain experimental;
their known blockers are recorded in [`docs/NOTEBOOK_ANALYSIS.md`](../../docs/NOTEBOOK_ANALYSIS.md).

### Check your understanding

1. Why is the ball radius `1/sqrt(c)` rather than `1` when `c!=1`?
2. Why does the Euclidean limit contain a factor of `2` under this convention?
3. **Coding exercise:** add a test that geodesic distances satisfy
   `d_c(gamma(0),gamma(t)) ~= t d_c(u,v)` at five values of `t`.


### Вводная информация

Рассмотрим бинарное дерево из семи вершин
`root -> {A,B} -> {A1,A2,B1,B2}`. Более глубокие уровни можно разместить на
больших радиусах Пуанкаре. Около границы доступно больше углового пространства,
а метрическое расстояние до границы расходится. Поэтому в обученном эмбеддинге
радиус может коррелировать с глубиной, а угол — разделять ветви. Это осторожная
интерпретация, а не инвариант: изометрия может сместить видимый центр, а функция
потерь может не кодировать глубину.

### Соглашение и область

**Блок соглашений.** Пусть `n>=1`, `c>0`, а секционная кривизна `K=-c`.
Векторы `x,u,v` заданы евклидовыми координатами в `R^n`; `||.||` и `<.,.>` —
евклидовы. Шар Пуанкаре и конформная метрика равны

```math
\mathbb D_c^n=\{x\in\mathbb R^n:c\lVert x\rVert^2<1\},
\qquad
\lambda_x^c=\frac{2}{1-c\lVert x\rVert^2},
\qquad
g_x^c=(\lambda_x^c)^2 I_n.
```

Координатный радиус равен `1/sqrt(c)`. Граница не входит в многообразие.
Расстояние равно

```math
d_c(u,v)=\frac{1}{\sqrt c}\mathrm{arcosh}\!\left(
1+\frac{2c\lVert u-v\rVert^2}
{(1-c\lVert u\rVert^2)(1-c\lVert v\rVert^2)}\right).
```

Все знаменатели положительны только для внутренних точек. В начале координат

```math
d_c(0,x)=\frac{2}{\sqrt c}\mathrm{artanh}(\sqrt c\lVert x\rVert).
```

Это следует из подстановки `u=0` и тождества
`arcosh((1+t^2)/(1-t^2))=2 artanh(t)` при `0<=t<1`.

### Сложение Мёбиуса и геодезические

Для внутренних `u,v` определим

```math
u\oplus_c v=
\frac{(1+2c\langle u,v\rangle+c\lVert v\rVert^2)u
{}+(1-c\lVert u\rVert^2)v}
{1+2c\langle u,v\rangle+c^2\lVert u\rVert^2\lVert v\rVert^2}.
```

Для `x!=0` и вещественного `t` умножение Мёбиуса на скаляр равно

```math
t\otimes_c x=
\frac{\tanh\!\left(t\mathrm{artanh}(\sqrt c\lVert x\rVert)\right)}
{\sqrt c\lVert x\rVert}x,
\qquad t\otimes_c0=0.
```

Геодезический отрезок от `u` до `v` можно параметризовать как

```math
\gamma_{u\to v}(t)=u\oplus_c\left(t\otimes_c((-u)\oplus_c v)\right),
\qquad 0\le t\le1.
```

В отличие от евклидовых хорд эти кривые обычно выглядят как дуги окружностей,
ортогональных граничной окружности. Диаметры — частный прямолинейный случай.

### Экспоненциальное и логарифмическое отображения в начале координат

Касательное пространство в начале координат — `T_0 D_c^n ~= R^n`. В принятом
координатном соглашении для ненулевых аргументов

```math
\exp_0^c(w)=
\frac{\tanh(\sqrt c\lVert w\rVert)}{\sqrt c\lVert w\rVert}w,
\qquad
\log_0^c(x)=
\frac{\mathrm{artanh}(\sqrt c\lVert x\rVert)}
{\sqrt c\lVert x\rVert}x.
```

Непрерывное значение в нуле равно нулю. Типы отображений:
`exp_0^c:T_0 D_c^n -> D_c^n` и `log_0^c:D_c^n -> T_0 D_c^n`.
Подстановка даёт `log_0^c(exp_0^c(w))=w` в точной арифметике.

### Предельный случай и численные ограничения

Поскольку `lambda_x^c -> 2` при `c -> 0+`, в этом соглашении

```math
\lim_{c\to0^+}d_c(u,v)=2\lVert u-v\rVert.
```

Множитель `2` не является ошибкой. Соглашение с евклидовой предельной метрикой
`I_n` иначе масштабирует расстояние. При реализации:

1. используйте числа с плавающей точкой и отклоняйте `c<=0`;
2. после обновлений проецируйте на радиус `(1-eps)/sqrt(c)`;
3. перед `artanh` ограничивайте `sqrt(c)||x||` значением меньше `1`;
4. ограничивайте аргумент `arcosh` снизу единицей;
5. тестируйте обычные и приграничные точки, желательно в `float64`.

**Распространённое заблуждение.** Хорда на рисунке Пуанкаре обычно не является
геодезической. Она является геодезической только на диаметре (либо после
перехода к координатам модели Клейна, где геодезические выглядят как хорды, но
метрические координаты отличаются).

### Реализация в репозитории

| Математический объект | Реализация в репозитории | Состояние |
|---|---|---|
| `D_c^n`, проекция | `inside_ball`, `project` в курируемом notebook | Детерминированные тесты; чистый запуск Colab пока не зафиксирован |
| `d_c` | `poincare_distance` | Проверены тождество, симметрия, рост у границы и евклидов предел |
| `oplus_c`, `otimes_c` | `mobius_add`, `mobius_scalar_mul` | Включены тесты концов геодезической |
| `exp_0^c`, `log_0^c` | `expmap0`, `logmap0` | Включён тест round-trip |
| Exp/log в общей точке | Пример отсутствует | Запланировано |
| Преобразования Лоренца/Клейна | Пример отсутствует | После проверки соглашений по источникам |



### Проверьте понимание

1. Почему при `c!=1` радиус шара равен `1/sqrt(c)`, а не `1`?
2. Почему в принятом соглашении евклидов предел содержит множитель `2`?
3. **Задание:** добавьте проверку
   `d_c(gamma(0),gamma(t)) ~= t d_c(u,v)` для пяти значений `t`.

