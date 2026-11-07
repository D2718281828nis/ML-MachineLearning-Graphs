# Notebook analysis / Анализ notebooks

This inventory explains the **code path**, not just the title. It is based on a
static inspection of cells and stored outputs; only the checks listed in the
commit summary establish current execution status. / Этот каталог описывает
**логику кода**, а не только названия. Он основан на статическом просмотре ячеек
и сохранённых outputs; актуальный статус запуска определяется только проверками,
указанными в итогах изменения.

## Curated examples / Учебные примеры

| Notebook | What the code does / Что делает код | Status and limits / Статус и ограничения |
|---|---|---|
| `examples/graph_learning_basics.ipynb` | Creates, inspects, and plots a small NetworkX graph. / Создаёт, исследует и рисует небольшой граф NetworkX. | Deterministic teaching example; see its in-notebook verification statement. / Детерминированный учебный пример; статус указан внутри. |
| `examples/hyperbolic_graph_basics.ipynb` | Computes Poincaré-ball distances and embeds a tree. / Вычисляет расстояния Пуанкаре и embedding дерева. | Numerical illustration, not a trained GNN. / Численный пример, не обученная GNN. |
| `examples/graph_connectivity_chains_circles.ipynb` | States the problem formally (formulas (1.1)–(1.16), shared with `theory/graph-connectivity/`), checks the chain model's predictions on the data, then explains, figure by figure, why connected components of a k = 2 k-NN graph separate the chain classes: local neighbourhoods, edge homophily, component sizes, h-hop reach, the choice of k, and a comparison with local, global and supervised (5-fold CV) methods. / Загружает четыре набора точек, измеряет гомофилию рёбер k-NN графа и сравнивает плотность, LOF и сглаживание с компонентами связности графа при k = 2; перебирает k. | Executed top to bottom on a clean CPU kernel (2026-11). Synthetic data: a teaching example, not a benchmark. Condensed from [ML-overlay-problem](https://github.com/D2718281828nis/ML-overlay-problem). / Выполнен целиком на CPU; synthetic-данные, учебный пример. |
| `examples/ai_medical_devices_graph_visualizations.ipynb` | Builds device/relationship graphs and several visualizations. / Строит граф медицинских устройств и визуализации. | Educational visualization; no outcome prediction. / Учебная визуализация без прогноза исходов. |
| `example Gated Graph Attention Network.ipynb` | Defines a gated, multi-head graph-attention layer in PyTorch/PyG. / Определяет gated multi-head graph-attention layer. | Pins wheel assumptions to a Colab CUDA/PyTorch combination; may need dependency updates. / Зависит от конкретной Colab CUDA/PyTorch-сборки. |
| `example_Interactive_Graph_Visualisation.ipynb` | Creates a five-node NetworkX graph and serializes an interactive view. / Создаёт граф из пяти вершин и интерактивное представление. | Uses browser/Colab display behavior. / Использует браузерный/Colab display. |

## Structured healthcare lab / Структурированная healthcare-лаборатория

The numbered notebooks are orchestration layers. They load JSON configuration
and call reusable, tested components from `healthcare-gnn-lab/src/healthcare_gnn`
rather than hiding model logic in cells. / Нумерованные notebooks являются
слоем orchestration: они загружают JSON-конфигурацию и вызывают переиспользуемые
компоненты из `src/`, не скрывая логику модели в ячейках.

| Notebook | Pipeline / Pipeline |
|---|---|
| `01_patient_similarity_mi.ipynb` | Patient-level split → train-only preprocessing → label-free similarity graph → logistic, GCN, and Poincaré/tangent comparison. / Split по пациентам → train-only preprocessing → граф без меток → сравнение logistic, GCN и Poincaré/tangent. |
| `02_ptbxl_ecg_graph.ipynb` | Demonstrates the 12-lead ECG graph/encoder interface and shared smoke runner. / Демонстрирует интерфейс графа 12 ECG-отведений, encoder и общий smoke runner. |
| `03_mimic_ehr_graph.ipynb` | Exercises temporal and phenotype guards for EHR plus the shared runner. / Проверяет временные и phenotype-ограничения EHR и общий runner. |
| `04_brain_hypergraph_optional.ipynb` | Applies an explicit incidence matrix/operator for an optional brain hypergraph adaptation. / Использует явную incidence-матрицу для brain-hypergraph адаптации. |

Synthetic runs test plumbing only. Real medical work requires access review,
patient grouping, temporal feature audit, nested validation, repeated seeds, and
external validation. / Synthetic-запуски проверяют только pipeline. Для реального
исследования нужны контроль доступа, split по пациентам, временной аудит
признаков, nested validation, разные seed и внешняя проверка.

### Medical/genomic explorations 05–07 / Медицинские и геномные эксперименты 05–07

| Notebook | What the cells do / Что делают ячейки | Reproducibility limits / Ограничения воспроизводимости |
|---|---|---|
| `05_genomic_population_gcn_exploratory.ipynb` | Loads genetic tabular data, constructs a k-NN sample graph, visualizes PCA/t-SNE/network layouts, trains a PyG GCN, and compares hidden widths. / Загружает генетическую таблицу, строит k-NN граф образцов, визуализирует PCA/t-SNE/сеть, обучает PyG GCN и сравнивает hidden width. | Data loading is notebook-specific and the original evaluation does not implement the lab's complete patient/group leakage contract. Population inference is sensitive and is not a clinical classifier. / Загрузка специфична, а оценка не реализует полный leakage-контракт лаборатории; population inference — чувствительная задача, не клинический классификатор. |
| `06_hyperbolic_tp53_sequences_exploratory.ipynb` | Encodes TP53 protein sequences, computes edit distances, creates k-NN adjacency, trains a custom hyperbolic model with an unsupervised hierarchy loss, clusters/plots embeddings, and compares Euclidean methods. / Кодирует TP53-последовательности, считает edit distance, строит k-NN adjacency, обучает custom hyperbolic model, кластеризует/рисует embeddings и сравнивает Euclidean methods. | Custom geometry and a stored cell error require independent numerical validation; example sequences are not clinical evidence. / Custom-геометрия и сохранённая ошибка ячейки требуют проверки; пример не является клиническим доказательством. |
| `07_vitiligo_image_clustering_exploratory.ipynb` | Downloads a composite vitiligo image, slices it into regions, derives image/segmentation quantities, and explores KL-divergence-based agglomerative grading. / Загружает composite-изображение витилиго, делит на области, вычисляет признаки/сегментацию и исследует agglomerative grading на основе KL divergence. | Uses a fixed image and Colab paths; it is an exploratory reproduction, not validated grading software. / Использует одно изображение и Colab-пути; это исследовательский пример, а не валидированная grading-система. |

These reviewed files now follow the numbered notebook layout, but the
`exploratory` suffix distinguishes them from smoke-enabled entry points 01–04.
Reimplementation should migrate reviewed data contracts and model
components into `src/` before any result is compared with the lab baselines. /
Файлы находятся рядом, но отделены от нумерованных notebooks. Перед сравнением с
baseline нужно перенести проверенные data contracts и компоненты моделей в `src/`.

## Research notebooks / Исследовательские notebooks

| Notebook | Code behavior / Поведение кода | Main blocker / Главное ограничение |
|---|---|---|
| `Graph from DataFrame tSNE.ipynb` | Loads the red/green table from `datasets/`, fits t-SNE once on all points, scores a class-balanced random forest on (X, Y) and on the t-SNE map with 5-fold CV, builds a label-free k-NN graph and reports its homophily. / Загружает таблицу из `datasets/`, строит t-SNE один раз, оценивает сбалансированный random forest 5-fold CV, строит k-NN граф без меток и считает гомофилию. | Re-run after the 2026-11 fixes (see the audit note in the notebook): the old graph used same-`Type` edges, which leak the label, and the old t-SNE "test set" was a separate embedding. Now F1 (red) ≈ 0.43 vs. the 0.239 baseline. / Перезапущен после исправлений: старые рёбра по `Type` давали утечку метки. |
| `Gromov-Waserstein graph clastering.ipynb` | Loads the same table and invokes the external Spectral-Gromov-Wasserstein repository for graph clustering. / Загружает таблицу и использует внешний Spectral-Gromov-Wasserstein repo. | Stored error (an interrupted run) and unpinned external code. The download now points at this repository; outputs are from the earlier run. / Сохранённая ошибка и незакреплённая внешняя версия; ссылка на данные исправлена. |
| `Gromov-Wasserstein distance Transport Task.ipynb` | Creates synthetic structured graphs and computes/visualizes a GW-style transport comparison. / Создаёт synthetic-графы и визуализирует GW transport comparison. | Demonstration rather than benchmark. / Демонстрация, не benchmark. |
| `Hyperbolic Graph Neural Network.ipynb` | Defines simplified hyperbolic linear/convolution operations, builds a coordinate k-NN graph, trains with a class-weighted loss on a stratified split, reports held-out F1, and plots the Poincaré representation. / Определяет упрощённые hyperbolic layers, обучает со взвешенной функцией потерь на стратифицированном split и рисует представление Пуанкаре. | Re-run after the 2026-11 fixes: the unweighted original predicted no minority node (F1 = 0) and scored accuracy on training nodes. Now held-out F1 ≈ 0.47 on one split (noisy). Simplified manifold operations. / Перезапущен после исправлений; геометрия упрощена. |
| `Poincare ball for Graph.ipynb` | Contrasts Euclidean and Poincaré positions and illustrates boundary-distance effects. / Сравнивает Euclidean/Poincaré позиции и эффект границы. | Visual intuition only. / Только визуальная интуиция. |
| `LLM Graph triplets visualisation.ipynb` | Calls Mistral to extract triples, turns them into NetworkX/PyVis graphs, and exports HTML. / Вызывает Mistral для triples, строит NetworkX/PyVis граф и экспортирует HTML. | Requires a user API key; never commit it. / Требует API key пользователя; ключ нельзя коммитить. |
| `ML task table PCA ICA tSNE DBSCAN AggClustering SVM.ipynb` | Generates several synthetic tabular cases and surveys PCA, Kernel PCA, ICA, t-SNE, DBSCAN, one-class/SVC, agglomerative clustering, and k-means. / Генерирует synthetic cases и сравнивает PCA, Kernel PCA, ICA, t-SNE, DBSCAN, SVM, agglomerative clustering и k-means. | Large exploratory notebook with partially executed cells and repeated state. / Большой notebook с частичным выполнением и повторно используемым state. |

## Utilities / Утилиты

| Notebook | Function / Назначение | Limits / Ограничения |
|---|---|---|
| `utilities/notebooks/erd_to_mermaid.ipynb` | Parses uploaded ERD XML and emits Mermaid text. / Разбирает ERD XML и формирует Mermaid. | Colab upload/widgets; test only with non-sensitive schemas. / Colab upload/widgets; используйте только нечувствительные схемы. |
| `utilities/notebooks/erd_graph_visualisation.ipynb` | Parses ERD entities/relations and creates static clustered NetworkX figures. / Создаёт статические clustered NetworkX-визуализации ERD. | Layout is heuristic and stored output is large. / Heuristic layout и большой output. |
| `utilities/notebooks/erd_interactive_graph_visualisation.ipynb` | Adds interactive ERD rendering and optional AI-assisted translation workflow. / Добавляет интерактивный ERD-rendering и опциональный AI-перевод. | Colab-only upload/display and external-service review may be required. / Требует Colab и проверки внешнего сервиса. |

## Audit method / Метод проверки

Run `python scripts/audit_notebooks.py`. It recursively validates JSON parsing,
reports cell/output/error counts, checks every known notebook classification,
and rejects root notebooks not prefixed by `demo` or `example`. It does **not**
execute notebook cells. / Команда рекурсивно проверяет JSON, считает ячейки,
outputs и ошибки, проверяет классификацию и запрещает в корне notebooks без
префикса `demo`/`example`. Ячейки при этом **не выполняются**.
