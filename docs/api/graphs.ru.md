# Визуализация данных («Графики») { #data-visualization-graphs }

Модуль `spectroview.api.graphs` позволяет программно создавать графики публикационного качества, с точностью до пикселя воспроизводящие штатное оформление рабочей области **Графики** (Graphs) в SPECTROview для всех 9 поддерживаемых стилей графиков. Он обращается ровно к тому же коду отрисовки, который использует виджет «Графики» в графическом интерфейсе (без отдельной повторной реализации и без приближений), поэтому графики, построенные из скрипта, полностью совпадают с тем, что рисует GUI: то же чередование цветов и маркеров, тот же расчёт 95%-го доверительного интервала (95% ДИ) и то же оформление для диаграмм «ящик с усами», точечных графиков, линий тренда, карт пластин и 2D-карт.

---

## Подготовка данных { #preparing-data }

Функции построения графиков принимают стандартный объект Pandas DataFrame — обычно это результат вызова `SpectraWorkspace.collect_results()` / `MapsWorkspace.collect_results()` или данные, загруженные с помощью `io.load_dataset()`.

```python
import pandas as pd
from spectroview.api import io

df = io.load_dataset("fit_results.csv")["fit_results"]
print(df.columns)
```

Каждая функция `plot_*` принимает необязательный параметр `ax=` (в этом случае график строится на переданных осях вместо создания новой фигуры) и возвращает объект `Axes`, на котором выполнена отрисовка, что позволяет самостоятельно компоновать многопанельные рисунки.

---

## Категориальные и XY-графики { #categorical-xy-plots }

### Статистические точечные графики { #statistical-point-plots }

Среднее значение с планками погрешностей 95%-го доверительного интервала, сгруппированное по категориальной переменной — удобно для сравнения параметров пиков при различных условиях эксперимента.

```python
import matplotlib.pyplot as plt
from spectroview.api import graphs

fig, ax = plt.subplots(figsize=(8, 6))
graphs.plot_point(df, x="Sample", y="peak_1_x0", hue="Condition",
                   join=False, dodge=True, title="Peak Shift by Process Condition", ax=ax)
plt.tight_layout()
plt.savefig("point_plot.png", dpi=300)
```

### Линейные графики { #line-plots }

Аналогично `plot_point`, но соединяет средние значения по каждому X линией с закрашенной доверительной полосой 95% ДИ — удобно для переменных с естественным упорядочением (например, глубина, время, доза).

```python
graphs.plot_line(df, x="Depth_nm", y="peak_1_ampli", hue="Condition", title="Amplitude vs Depth")
```

### Столбчатые диаграммы { #bar-plots }

```python
graphs.plot_bar(df, x="Sample", y="peak_1_ampli", hue="Condition", show_error_bar=True)
```

### Диаграммы «ящик с усами» { #box-plots }

```python
graphs.plot_box(df, x="Sample", y="peak_1_fwhm", hue="Condition", title="FWHM Distribution Analysis")
```

### Гистограммы { #histograms }

```python
graphs.plot_histogram(df, x="peak_1_fwhm", hue="Condition", bins=20, kde=True)
```

### Диаграммы рассеяния и линии тренда { #scatter-plots-trendlines }

```python
graphs.plot_scatter(df, x="peak_1_ampli", y="peak_2_ampli", hue="Condition",
                     title="Peak 1 vs Peak 2 Amplitude Correlation")

graphs.plot_trendline(df, x="peak_1_x0", y="peak_1_fwhm", order=1, hue="Condition",
                       title="Correlation: FWHM vs Peak Position")
```

---

## Пространственные графики { #spatial-plots }

Эти функции принимают табличный (*tidy*) DataFrame с одной строкой на каждую точку измерения (x, y) и числовым столбцом значений `z` — именно в таком формате возвращает данные `MapsWorkspace.collect_results()`.

### 2D-карты { #2d-maps }

Быстрая и точная тепловая карта на собственной регулярной сетке измерения (растровое сканирование / 2D-карты).

```python
graphs.plot_2dmap(df, x="X", y="Y", z="ampli_Si", cmap="jet", title="Amplitude Map")
```

### Карты пластин { #wafer-maps }

Интерполированная тепловая карта в пределах круглого контура пластины с возможностью отображения сводной статистики непосредственно на графике.

```python
graphs.plot_wafer(df, x="X", y="Y", z="ampli_Si", wafer_size=300.0, show_stats=True)
```

---

## Рецепты графиков { #plot-recipes }

Рецепт графиков (*plot recipe*) — это именованный, многократно используемый набор конфигураций графиков (каждая из которых имеет формат словаря, возвращаемого методом `MGraph.save()` в графическом интерфейсе), сохраняемый в виде JSON-файла в каталоге — это тот же механизм, который использует функция **Сохранить рецепт** (Save Plot Recipe) в GUI. (Не путать с *шаблоном стиля*, который сохраняет только визуальное оформление без привязок к данным — см. `spectroview/model/graphs/m_style_template_store.py`.)

```python
from spectroview.api import graphs

configs = [
    {"plot_style": "scatter", "x": "peak_1_ampli", "y": ["peak_2_ampli"], "z": "Condition"},
    {"plot_style": "box", "x": "Sample", "y": ["peak_1_fwhm"], "z": "Condition"},
]
recipe_id = graphs.save_plot_recipe("./recipes", "Standard QC Plots", configs)

# List and reload later
for summary in graphs.list_plot_recipes("./recipes"):
    print(summary["name"], summary["graph_count"])

configs = graphs.load_plot_recipe("./recipes", recipe_id)
graphs.delete_plot_recipe("./recipes", recipe_id)
```
