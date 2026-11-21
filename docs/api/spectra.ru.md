# Рабочая область «Спектры» { #spectra-workspace }

В этом разделе показано, как работать с дискретными 1D-спектрами: выполнять их загрузку, предобработку и аппроксимацию с помощью векторизованного движка пакетной аппроксимации (Vectorized Batch Fit, VBF) — как через сессию `SpectraWorkspace` с сохранением состояния (рекомендуемый способ), так и с помощью низкоуровневых функций для работы с массивами, на которых она основана.

---

## Рабочие области с сохранением состояния { #stateful-workspaces }

`spectroview.api.workspace.SpectraWorkspace` повторяет структуру рабочей области «Спектры» в графическом интерфейсе (GUI): он хранит ваши спектры, отслеживает состояние предобработки и модели аппроксимации для каждого спектра и читает/записывает точно такой же формат файлов `.spectra`, какой создаёт кнопка **Сохранить проект** (Save work) в интерфейсе. Рабочая область, полностью созданная в скрипте, открывается непосредственно в GUI, а файл `.spectra`, сохранённый из GUI, загружается напрямую здесь.

```python
from spectroview.api import workspace, fitting

ws = workspace.SpectraWorkspace()
ws.load_files(["sample_a.wdf", "sample_b.wdf", "sample_c.txt"])
print(ws)  # SpectraWorkspace(3 spectra: ['sample_a', 'sample_b', 'sample_c'])

# Preprocess every spectrum in the workspace
ws.crop(range_min=400.0, range_max=800.0)
ws.set_baseline({"mode": "arpls", "smoothing_factor": 1e5})
ws.subtract_baseline()

# Attach a fit model and fit
fit_model = fitting.build_fit_model(
    peaks=[
        {"model": "Lorentzian",
         "x0": {"value": 520.0, "min": 515.0, "max": 525.0},
         "ampli": {"value": 1000.0, "min": 0.0, "max": 1e9},
         "fwhm": {"value": 3.0, "min": 1.0, "max": 10.0}},
    ]
)
ws.set_fit_model(fit_model)
ws.fit()

df = ws.collect_results()   # pandas DataFrame, one row per spectrum
print(df)

# Save — this file opens directly in the SPECTROview GUI
ws.save("session.spectra")
```

Чтобы повторно загрузить сессию позже (созданную через этот API или сохранённую из GUI):

```python
ws2 = workspace.SpectraWorkspace.load("session.spectra")
```

Каждый метод предобработки или аппроксимации принимает необязательный список `names=[...]`, чтобы применить операцию только к подмножеству спектров вместо всех сразу, например `ws.crop(range_min=400, range_max=800, names=["sample_a"])`.

!!! note "Устаревшие файлы"
    Очень старые файлы `.spectra` (формат v1, предшествующий текущему формату сохранения на базе ZIP) поддерживаются только в GUI. При их открытии `SpectraWorkspace.load()` вызывает исключение `WorkspaceError` — откройте такой файл один раз в графическом интерфейсе и пересохраните его, чтобы обновить до актуального формата.

---

## Загрузка данных { #loading-data }

Модуль `spectroview.api.io` предоставляет унифицированные функции для загрузки спектров из различных форматов (WDF, SPC, L6S, NGS, CSV, TXT, DAT) — именно их `SpectraWorkspace.load_files()` использует внутри себя, и они также удобны сами по себе для низкоуровневой работы с массивами, описанной ниже.

```python
from spectroview.api import io
import numpy as np

spectra_dict = io.load_spectra("my_sample_spectra.wdf")

# Each item in the list is one spectrum (WDF series files can contain many)
items = spectra_dict["items"]
print(f"Number of spectra loaded: {len(items)}")

# Each item has: "name", "x0" (wavenumber axis), "y0" (intensities), "metadata"
first = items[0]
print("Wavenumber axis (x0):", first["x0"].shape)   # shape (M,)
print("Intensities (y0):    ", first["y0"].shape)   # shape (M,)
print("Laser Wavelength:", first["metadata"].get("Laser Wavelength (nm)"))
```

### Пакетная загрузка нескольких файлов { #loading-multiple-files-as-a-batch }

Чтобы работать с набором спектров одновременно как с единой 2D-матрицей (вместо `SpectraWorkspace`), используйте `io.load_spectra_to_matrix()`:

```python
import glob

file_paths = sorted(glob.glob("./measurements/*.wdf"))
data = io.load_spectra_to_matrix(file_paths)

x     = data["x"]      # float64[M]   — shared wavenumber axis
Y     = data["Y"]      # float64[N,M] — intensity matrix (N spectra)
names = data["names"]  # list[str]    — spectrum names
```

!!! note
    Функция `load_spectra_to_matrix()` интерполирует все спектры на ось X первого файла, если их спектральные оси немного различаются.

---

## Предобработка { #preprocessing }

Для прямого управления на уровне массивов без использования `SpectraWorkspace` модуль `spectroview.api.preprocessing` предоставляет те же операции обрезки, вычитания базовой линии и нормировки в виде функций без сохранения состояния, работающих с массивами `(x, Y)`.

```python
from spectroview.api import preprocessing

# Crop to the region of interest
x_crop, Y_crop = preprocessing.crop_spectra(x, Y, range_min=400.0, range_max=800.0)

# Baseline subtraction (same algorithms as the GUI, e.g. arpls, airpls, asls, Linear, Polynomial)
baseline_config = {"mode": "arpls", "smoothing_factor": 1e5}
Y_corrected, Y_baseline = preprocessing.subtract_baseline(x_crop, Y_crop, baseline_config)

# Normalize each spectrum to its own maximum
Y_norm = preprocessing.normalize_spectra(Y_corrected)
```

---

## Пакетная аппроксимация { #batch-fitting }

Главное преимущество SPECTROview — его векторизованный движок пакетной аппроксимации (VBF), способный аппроксимировать тысячи спектров за доли секунды. Модуль `spectroview.api.fitting` открывает к нему прямой доступ для сценариев работы с массивами (вызов `SpectraWorkspace.fit()` использует его внутри себя по каждому спектру).

### Шаг 1: Определение или загрузка модели аппроксимации { #step-1-define-or-load-the-fit-model }

**Вариант А: Загрузка шаблона, экспортированного из GUI SPECTROview (или сохранённого через этот API)**

```python
from spectroview.api import fitting

fit_model = fitting.load_fit_model_template("my_gui_model.json")
```

**Вариант Б: Построение модели с помощью `fitting.build_fit_model()`**

```python
fit_model = fitting.build_fit_model(
    peaks=[
        {"model": "Lorentzian",
         "x0": {"value": 520.0, "min": 515.0, "max": 525.0},
         "ampli": {"value": 1000.0, "min": 0.0, "max": 1e9},
         "fwhm": {"value": 3.0, "min": 1.0, "max": 10.0}},
        {"model": "Gaussian",
         "x0": {"value": 600.0, "min": 580.0, "max": 620.0},
         "ampli": {"value": 500.0, "min": 0.0, "max": 1e9},
         "fwhm": {"value": 15.0, "min": 5.0, "max": 50.0}},
    ]
)
```

Словарь каждого параметра пика принимает ключи `value`, `min`, `max`, а также необязательные ключи `vary` (логическое значение, по умолчанию `True`) или `fix` (противоположный ему по смыслу) и `expr` (строковое выражение, связывающее этот параметр с другим).

### Шаг 2: Выполнение аппроксимации { #step-2-execute-the-fit }

```python
fit_params = {"xtol": 1e-4, "ftol": 1e-4, "max_ite": 500}

results = fitting.fit_batch(x_crop, Y_corrected, fit_model, fit_params=fit_params)

print(f"Success rate: {results['success'].mean() * 100:.1f}%")
print(f"Average R-squared: {results['r_squared'].mean():.4f}")

fitted_params_matrix = results["params"]        # shape (N, K)
parameter_names      = results["param_names"]   # e.g. ['P1_x0', 'P1_fwhm', ...]
```

По умолчанию `fit_batch` вычисляет веса для аппроксимации из `fit_params` точно так же, как это делает GUI перед каждым расчётом: точки с отрицательной интенсивностью исключаются (если только `fit_params["fit_negative"]` не равно `True`), а также исключаются точки ниже порога шума, задаваемого параметром `fit_params["coef_noise"]`. Передайте `auto_weights=False`, чтобы аппроксимировать все точки с одинаковым весом, или задайте собственный массив `weights`.

Если у вас есть матрица исходных спектров, к которой сначала нужно применить шаги обрезки и коррекции базовой линии из модели аппроксимации, используйте `fitting.apply_fit_model(x, Y, fit_model)` — эта функция объединяет все три этапа (обрезку, вычитание базовой линии и аппроксимацию) в одном вызове и дополнительно возвращает фактически использованные при аппроксимации массивы `x`/`Y`.

### Шаг 3: Экспорт результатов { #step-3-export-results }

```python
import pandas as pd
from spectroview.api import io

df_results = pd.DataFrame(results["params"], columns=results["param_names"])
df_results["R_squared"]   = results["r_squared"]
df_results["Fit_Success"] = results["success"]

io.export_results(df_results, "my_fit_results.csv")
```

### Сохранение модели аппроксимации для повторного использования { #saving-a-fit-model-for-reuse }

```python
fitting.save_fit_model_template(fit_model, "my_model.json")

# List and reload templates from a folder later
names = fitting.list_fit_model_templates("./models")
fit_model = fitting.load_fit_model_template("./models/my_model.json")
```
