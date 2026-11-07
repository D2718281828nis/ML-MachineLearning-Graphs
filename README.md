# Graph Machine Learning / Машинное обучение на графах

A collection of educational examples, exploratory notebooks, and a structured,
leakage-aware healthcare GNN laboratory. / Коллекция учебных примеров,
исследовательских notebooks и структурированной лаборатории healthcare GNN с
контролем утечек данных.

> **Research notice / Важно:** stored outputs in exploratory notebooks show prior
> experiments; they are not proof of a clean reproducible run or clinical
> validity. / Сохранённые результаты exploratory-notebooks не доказывают
> воспроизводимость или клиническую применимость.

## Start here / С чего начать

| Path / Раздел | English | Русский |
|---|---|---|
| [`examples/`](examples/) | Curated, small learning notebooks | Небольшие учебные notebooks |
| [Case study: overlay problem](#case-study-overlay-problem) | When a graph beats local methods (chains and circles) | Когда граф лучше локальных методов (цепочки и окружности) |
| [`healthcare-gnn-lab/`](healthcare-gnn-lab/README.md) | Tested healthcare pipeline and task entry points | Тестируемый healthcare pipeline и входные notebooks |
| [`research-notebooks/`](research-notebooks/) | Preserved exploratory graph-ML work | Сохранённые исследовательские эксперименты |
| [`utilities/notebooks/`](utilities/notebooks/) | ERD conversion and visualization tools | Утилиты преобразования и визуализации ERD |
| [`docs/NOTEBOOK_ANALYSIS.md`](docs/NOTEBOOK_ANALYSIS.md) | What every notebook does and its limitations | Назначение и ограничения каждого notebook |
| [`theory/hyperbolic-graphs/`](theory/hyperbolic-graphs/README.md) | Bilingual hyperbolic-graph theory | Двуязычная теория гиперболических графов |
| [`theory/graph-connectivity/`](theory/graph-connectivity/README.md) | Problem statement and formulas: graph connectivity vs. local methods | Постановка задачи и формулы: связность графа против локальных методов |

The repository root intentionally contains only notebooks whose filenames begin
with `example` or `demo`. / В корне намеренно оставлены только notebooks, имена
которых начинаются с `example` или `demo`.

## Repository map / Структура репозитория

```text
.
├── example*.ipynb              # root-level demos / корневые примеры
├── examples/                   # curated tutorials / учебные материалы
├── research-notebooks/         # non-medical exploratory work / исследования
├── utilities/notebooks/        # ERD tools / утилиты ERD
├── healthcare-gnn-lab/
│   ├── notebooks/              # numbered task entry points and explorations
│   ├── src/healthcare_gnn/     # reusable implementation
│   ├── configs/ and tests/
│   └── README.md               # bilingual lab guide / руководство
├── datasets/                   # small public example assets (see datasets/README.md)
├── theory/hyperbolic-graphs/   # EN/RU theory
├── theory/graph-connectivity/  # problem statement and formulas for the chain example
└── scripts/audit_notebooks.py  # JSON/inventory/root-policy audit
```

## Notebook groups / Группы notebooks

### Root examples / Примеры в корне

- `example Gated Graph Attention Network.ipynb` implements a custom gated
  multi-head attention layer. / Реализует gated multi-head attention layer.
- `example_Interactive_Graph_Visualisation.ipynb` builds a small NetworkX graph
  and renders it interactively. / Строит небольшой граф NetworkX и создаёт
  интерактивную визуализацию.

Curated tutorials in [`examples/`](examples/) / Учебные notebooks в `examples/`:

| Notebook | Topic / Тема |
|---|---|
| [`graph_learning_basics.ipynb`](examples/graph_learning_basics.ipynb) | Build, inspect and plot a small graph / Построение и анализ небольшого графа |
| [`hyperbolic_graph_basics.ipynb`](examples/hyperbolic_graph_basics.ipynb) | Poincaré-ball distances, tree embedding / Расстояния Пуанкаре, embedding дерева |
| [`graph_connectivity_chains_circles.ipynb`](examples/graph_connectivity_chains_circles.ipynb) | When a k-NN graph beats local methods ([case study](#case-study-overlay-problem)) / Когда k-NN граф лучше локальных методов |
| [`ai_medical_devices_graph_visualizations.ipynb`](examples/ai_medical_devices_graph_visualizations.ipynb) | Graph visualizations of AI medical devices / Визуализация графа медицинских AI-устройств |

### Healthcare / Здравоохранение

Medical and genomic notebooks now live under
[`healthcare-gnn-lab/notebooks/`](healthcare-gnn-lab/notebooks/). The numbered
notebooks `01`–`04` are thin entry points into tested code in `src/`; reviewed
exploratory studies are numbered `05`–`07` in the same folder. / Медицинские и
геномные notebooks находятся в `healthcare-gnn-lab/notebooks/`: notebooks
`01`–`04` используют тестируемый код из `src/`, а исследовательские работы
пронумерованы `05`–`07` в той же папке.

The exploratory genomic GCN, protein-sequence hyperbolic GNN, and vitiligo image
clustering notebooks are exploratory adaptations, **not** diagnostic tools. /
Exploratory-notebooks с genomic GCN, гиперболической GNN для белковых
последовательностей и кластеризацией изображений витилиго являются
исследовательскими примерами, **не** диагностическими средствами.

### Research and utilities / Исследования и утилиты

General graph construction, Gromov–Wasserstein, hyperbolic GNN, LLM triplet, and
classical-ML surveys are in [`research-notebooks/`](research-notebooks/). ERD to
Mermaid and ERD visualization notebooks are in
[`utilities/notebooks/`](utilities/notebooks/). / Общие эксперименты перенесены
в `research-notebooks/`, а ERD-конвертеры и визуализаторы — в
`utilities/notebooks/`.

See the [bilingual notebook analysis](docs/NOTEBOOK_ANALYSIS.md) for code flow,
data dependencies, and known blockers. / Подробный двуязычный разбор кода,
зависимостей и ограничений находится в
[анализе notebooks](docs/NOTEBOOK_ANALYSIS.md).

## Case study: the overlay problem / Кейс: задача наложения
<a id="case-study-overlay-problem"></a>

**Repository / Репозиторий:** [D2718281828nis/ML-overlay-problem](https://github.com/D2718281828nis/ML-overlay-problem) ·
**interactive site / интерактивный сайт:** <https://d2718281828nis.github.io/ML-overlay-problem/> ·
**illustrated explanation / объяснение в картинках:** [illustrations/README.md](https://github.com/D2718281828nis/ML-overlay-problem/blob/main/illustrations/README.md) ·
**here / здесь:** [`examples/graph_connectivity_chains_circles.ipynb`](examples/graph_connectivity_chains_circles.ipynb) (eight figures with explanations / восемь рисунков с объяснениями) ·
**theory / теория:** [`theory/graph-connectivity/`](theory/graph-connectivity/README.md) (problem statement, 16 formulas / постановка задачи, 16 формул)

Every row of [`datasets/df_red_green_overlay.csv`](datasets/df_red_green_overlay.csv)
is a 2-D point coloured red (227, 13.6%) or green (1,445). The task is to recover
the colour from X, Y alone. The case study grew out of this repository's red/green
notebooks. It runs 20+ clustering, density, embedding, graph and supervised methods
on four datasets, scores them honestly, and has an in-browser lab where every method
runs live. / Каждая строка — точка на плоскости, красная или зелёная; цвет нужно
восстановить только по X, Y. Кейс вырос из red/green notebooks этого репозитория:
более 20 методов на четырёх датасетах, честная оценка и интерактивная лаборатория в
браузере.

![Chains and circles datasets](https://raw.githubusercontent.com/D2718281828nis/ML-overlay-problem/main/illustrations/01_datasets.png)

**Main results / Главные результаты** (F1 for red; reproduced in the example notebook /
F1 для красного класса, воспроизводится в примере):

| Dataset / Датасет | All red (baseline) | Best local method, no labels | Graph: connected components (k = 2), no labels |
|---|---|---|---|
| original / исходный | 0.239 | 0.40 (k-NN sparsity) | 0.13 (≈ random) |
| ring / кольцо | 0.235 | 0.62 (k-NN density) | 0.26 (≈ random) |
| **chains / цепочки** | 0.239 | 0.18 (k-NN density) | **1.000** |
| **circles / окружности** | 0.239 | 0.82 (k-NN sparsity) | **1.000** |

On the walk chains even supervised methods lose: graph label spreading 0.78, a
k-NN classifier 0.74, polynomial logistic regression 0.32 (5-fold CV). / На цепочках
проигрывают даже методы с учителем.

**When does a graph help? / Когда помогает граф?** A graph method beats local methods
only when (a) the edges are **homophilous** (neighbours share the class) and (b) the
class depends on **long-range structure** (paths, components) rather than on local
density. In the chain dataset every point looks the same locally (two neighbours
12 px away), and only the chain length differs (red ~38 points, green 12–20).
Connected components of a k = 2 k-NN graph follow each chain to its end, so they
separate the colours perfectly. On the original data only 33% of a red node's edges
lead to red, so graphs (GCN, label propagation, hyperbolic GCN) at best match a plain
k-NN classifier. / Граф выигрывает, только если (a) рёбра гомофильны и (b) класс
зависит от дальней структуры, а не от локальной плотности. В цепочках все точки
локально одинаковы, различается только длина цепочки, и компоненты связности
k-NN графа при k = 2 разделяют цвета идеально. На исходных данных лишь 33% рёбер
красной вершины ведут к красным, и граф не помогает.

**What it changed here / Что исправлено здесь** (2026-11):

- [`research-notebooks/Graph from DataFrame tSNE.ipynb`](research-notebooks/Graph%20from%20DataFrame%20tSNE.ipynb)
  built its graph from same-`Type` edges. `Type` is the colour, so the graph leaked
  the answer. Edges are now a label-free k-NN graph, t-SNE is fitted once, and the
  classifier is class-balanced and cross-validated. / Рёбра по `Type` давали утечку
  метки; теперь k-NN граф без меток и честная кросс-валидация.
- [`research-notebooks/Hyperbolic Graph Neural Network.ipynb`](research-notebooks/Hyperbolic%20Graph%20Neural%20Network.ipynb)
  predicted no red point (accuracy 0.86, F1 = 0). It now trains with a
  class-weighted loss and reports F1 on held-out nodes only. / Модель не находила ни
  одной красной точки; теперь взвешенная функция потерь и оценка на отложенных
  вершинах.
- The chain, circle and ring datasets are in [`datasets/`](datasets/README.md),
  and all red/green notebooks load data from this repository. / Датасеты добавлены в
  `datasets/`, все notebooks читают данные из этого репозитория.

## Run / Запуск

For lightweight notebooks / Для простых notebooks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy pandas scipy scikit-learn matplotlib networkx
jupyter lab
```

Healthcare lab / Healthcare-лаборатория:

```bash
cd healthcare-gnn-lab
python -m pip install -e '.[test,data]'
pytest
healthcare-gnn --config configs/mi_smoke.json --output reports/mi_smoke.json
```

Audit notebook JSON and placement / Проверка JSON и расположения notebooks:

```bash
python scripts/audit_notebooks.py
```

Research notebooks have notebook-specific dependencies, external downloads, and
occasionally Colab-only APIs. Read the analysis before running them. /
Исследовательские notebooks имеют собственные зависимости, внешние загрузки и
иногда требуют Google Colab. Перед запуском прочитайте анализ.

## Contributing / Как внести вклад

1. Put only `demo*.ipynb` or `example*.ipynb` in the root; place reusable
   tutorials in `examples/`, healthcare work in `healthcare-gnn-lab/`, and tools
   in `utilities/`. / В корне оставляйте только `demo*` или `example*`.
2. Use descriptive `snake_case` filenames for new notebooks. / Используйте
   понятные имена в `snake_case`.
3. State objectives, prerequisites, expected runtime, data provenance, and
   verification status. / Укажите цели, зависимости, время, происхождение данных
   и статус проверки.
4. Set seeds, avoid absolute paths and secrets, and fit preprocessing on training
   data only. / Фиксируйте seed, не используйте абсолютные пути и секреты,
   обучайте preprocessing только на train.
5. Validate JSON, run relevant tests, and never claim clinical performance from
   synthetic data. / Проверяйте JSON и тесты; не делайте клинических выводов по
   synthetic-данным.

## License / Лицензия

Distributed under [`LICENSE`](LICENSE). / Распространяется на условиях
[`LICENSE`](LICENSE).
