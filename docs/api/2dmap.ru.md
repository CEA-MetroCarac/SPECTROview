# Рабочая область «Карты» и 2D-анализ { #maps-workspace-2d-analysis }

В этом разделе показано, как обрабатывать данные гиперспектральных карт (например, рамановских карт пластин): загружать данные, выполнять аппроксимацию каждого пикселя, строить тепловые карты и линейные профили, а также проводить многомерный анализ (MVA) — с помощью сессии `MapsWorkspace` с сохранением состояния (рекомендуемый способ), а для MVA — с помощью низкоуровневых функций для работы с массивами.

---

## Рабочие области с сохранением состояния { #stateful-workspaces }

Класс `spectroview.api.workspace.MapsWorkspace` воспроизводит рабочую область «Карты» графического интерфейса: по одной карте на каждый загруженный файл, причём все пиксели карты используют общую модель аппроксимации, а идентификаторы пикселей задаются в виде `f"{map_name}_({x}, {y})"` (или `f"{map_name}_({x}, {y})_{k}"`, если карты с временным разрешением содержат повторные сканирования в одной и той же координате `(x, y)`), в точности как в GUI. Он читает и записывает тот же формат файлов `.maps`, который создаётся кнопкой **Сохранить проект** (Save work) в графическом интерфейсе.

```python
from spectroview.api import workspace, fitting

ws = workspace.MapsWorkspace()          # or MapsWorkspace(map_type="wafer_300mm") for wafer maps
[map_name] = ws.load_files(["2Dmap_Si.txt"])

fit_model = fitting.load_fit_model_template("fit_model_Si.json")
ws.set_fit_model(fit_model, names=[map_name])
ws.fit(map_names=[map_name])            # one vectorized fit call over every pixel in the map

df = ws.collect_results()               # one row per pixel, includes X/Y coordinates
print(df.head())

# Build a heatmap of a fitted parameter
xi, yi, zi = ws.get_heatmap(map_name, "ampli_Si")

import matplotlib.pyplot as plt
plt.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin="lower", cmap="jet")
plt.colorbar(label="Amplitude")
plt.savefig("heatmap.png", dpi=200)

# Extract a line profile between two points on the heatmap
profile = ws.extract_profile(map_name, "ampli_Si", point1=(xi.min(), yi.min()), point2=(xi.max(), yi.max()))
print(profile)  # columns: X, Y, distance, values

# Save — this file opens directly in the SPECTROview GUI
ws.save("session.maps")
```

Метод `get_heatmap(map_name, value_col, ...)` принимает `value_col="Intensity"` или `"Area"` (вычисляются непосредственно по исходным интенсивностям, аппроксимация не требуется) либо любой столбец с параметром аппроксимации, присутствующий в `df_fit_results` (предварительно вызовите `collect_results()`). Для карт пластин (`map_type != "2Dmap"`) сетка строится методом интерполяции нерегулярно расположенных точек по радиусу пластины, определяемому параметром `map_type`; для `"2Dmap"` выполняется быстрое точное преобразование (pivot) на регулярной сетке самого измерения.

Повторно загрузить сессию позже (созданную с помощью данного API или сохранённую в графическом интерфейсе):

```python
ws2 = workspace.MapsWorkspace.load("session.maps")
```

Класс `MapsWorkspace` наследует все методы предобработки из `SpectraWorkspace` (`crop`, `set_baseline`, `subtract_baseline`, `normalize`, `reinit`) — каждый из них применяется сразу ко всем пикселям выбранной карты или карт.

---

## Прямая загрузка данных карты { #loading-map-data-directly }

Для управления данными на уровне массивов без `MapsWorkspace` функция `spectroview.api.io.load_map()` возвращает исходный DataFrame карты (именно её внутренне использует `MapsWorkspace.load_files()`):

```python
from spectroview.api import io
import numpy as np

# WDF, SPC, L6M, and NGC files return (DataFrame, metadata_dict); TXT/CSV return a DataFrame only.
result = io.load_map("my_wafer_map.wdf")
df_map, metadata = result if isinstance(result, tuple) else (result, {})
```

Первые два столбца возвращаемого DataFrame всегда содержат пространственные координаты (`X`, `Y`), а все остальные столбцы — значения волновых чисел (в виде строк):

```python
x_coords = df_map["X"].to_numpy(dtype=float)
y_coords = df_map["Y"].to_numpy(dtype=float)

wn_cols  = [c for c in df_map.columns if c not in ("X", "Y")]
x_axis   = np.array([float(c) for c in wn_cols])
Y_matrix = df_map[wn_cols].to_numpy(dtype=float)
```

Матрицу `Y_matrix` можно передать непосредственно в функцию `spectroview.api.fitting.fit_batch()` — см. раздел [Рабочая область «Спектры»: пакетная аппроксимация](spectra.md#batch-fitting).

---

## Многомерный анализ (MVA) { #multivariate-analysis-mva }

Многомерный анализ позволяет исследовать большие сложные наборы данных, не требуя предварительного знания положений пиков. Модуль `spectroview.api.analysis` предоставляет методы PCA и NMF, работающие с любой матрицей формы `(n_spectra, n_wavenumbers)` — например, с приведённой выше `Y_matrix` или с `Y0` из базового хранилища `MapsWorkspace`.

### Метод главных компонент (PCA) { #principal-component-analysis-pca }

```python
from spectroview.api import analysis, preprocessing

Y_norm = preprocessing.normalize_spectra(Y_matrix)
pca_result = analysis.pca(Y_norm, n_components=3, center=True)

print("Explained Variance Ratios:", pca_result.explained_variance_ratio)
scores   = pca_result.scores     # (n_spectra, n_components)
loadings = pca_result.loadings   # (n_components, n_wavenumbers)
```

### Неотрицательная матричная факторизация (NMF) { #non-negative-matrix-factorization-nmf }

NMF накладывает условие строгой положительности на компоненты и факторы (scores), благодаря чему зачастую позволяет получать более физически интерпретируемые спектры «чистых компонент», чем PCA.

```python
nmf_result = analysis.nmf(Y_norm, n_components=3, max_iter=1000)
scores_w   = nmf_result.W
loadings_h = nmf_result.H
```

### Ошибка восстановления { #reconstruction-error }

```python
errors = analysis.reconstruction_error(Y_norm, pca_result.scores, pca_result.loadings, pca_result.mean_spectrum)
# errors: (n_spectra,) per-spectrum L2 residual -- useful for flagging outlier spectra
```

### Экспорт результатов MVA { #exporting-mva-results }

```python
import pandas as pd

df_pca = pd.DataFrame({"X": x_coords, "Y": y_coords})
for i in range(pca_result.n_components):
    df_pca[f"PCA_Score_PC{i+1}"] = scores[:, i]

df_pca.to_csv("map_with_pca_scores.csv", index=False)
```

Чтобы визуализировать `df_pca` в виде тепловой карты так же, как это делает `MapsWorkspace.get_heatmap()`, используйте функцию `spectroview.model.spectra.heatmap.build_heatmap_grid(x_coords, y_coords, scores[:, 0], map_type="2Dmap")`.
