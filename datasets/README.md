# Datasets / Наборы данных

Small public example assets used by the notebooks. / Небольшие открытые файлы для
notebooks.

## Red/green point clouds / Красные и зелёные точки

Every file has the columns `X`, `Y`, `Color` (`red` / `light_green`) and `Type`
(`point` / `cross`). **`Type` is the colour under another name** (point = red,
cross = green): use it only as the target, never as a feature or to build graph
edges, or the label leaks. / **`Type` дублирует цвет**: используйте его только как
целевую переменную, но не как признак и не для построения рёбер графа.

| File | Points (red) | Red class | Green class |
|---|---|---|---|
| `df_red_green_overlay.csv` | 1,672 (227) | original data: the same distribution as green / исходные данные, то же распределение | – |
| `df_red_green_distinct.csv` | 1,704 (227) | a noisy ring, R = 350, σ = 15 px / зашумлённое кольцо | uniform over the original footprint / равномерно |
| `df_red_green_chains.csv` | 1,672 (227) | 6 long random-walk chains (~38 points) / 6 длинных цепочек | ~95 short chains (12–20 points) / ~95 коротких цепочек |
| `df_red_green_circles.csv` | 1,671 (227) | 6 large closed circles (~38 points) / 6 больших окружностей | ~90 small circles (12–20 points) / ~90 малых окружностей |

`red_green_overlay.png` is a picture of the original data.

**Provenance / Происхождение.** The three synthetic files are copied unchanged from
[ML-overlay-problem](https://github.com/D2718281828nis/ML-overlay-problem) and
were made by its `make_distinct_dataset.py`:

```bash
python make_distinct_dataset.py                                          # distinct (ring)
python make_distinct_dataset.py --green chains --red chains              # chains
python make_distinct_dataset.py --green chains --red chains --chain-shape circle --seed 5   # circles
```

Used by / Где используются:
[`examples/graph_connectivity_chains_circles.ipynb`](../examples/graph_connectivity_chains_circles.ipynb)
(all four) and the red/green notebooks in
[`research-notebooks/`](../research-notebooks/) (original data only).

## Other files / Другие файлы

- `DB_MIS_REPORT - DB_MIS_REPORT - dbo.txt`: an example ERD schema export for the
  notebooks in [`utilities/notebooks/`](../utilities/notebooks/). / Пример выгрузки
  ERD-схемы для утилит.
