# Расширение SPECTROview { #extending-spectroview }

В SPECTROview нет системы плагинов или точек входа — в приложении отсутствует реестр для подключения пользовательского кода. Вместо этого расширяемость основана на двух файловых механизмах, которые API предоставляет напрямую. Если вы ищете ответ на вопрос «как добавить свой X в SPECTROview», почти всегда подойдёт один из них.

---

## Пользовательские методы базовой линии { #custom-baseline-methods }

Оценка базовой линии (`spectroview.api.preprocessing.subtract_baseline`, а также `set_baseline` в каждой рабочей области) выбирает алгоритм по строковому имени в `config["mode"]`. Помимо встроенных режимов `None`, `"Linear"`, `"Polynomial"` и `"arpls"`, **любое имя метода `Baseline` из [pybaselines](https://pybaselines.readthedocs.io/) работает «из коробки»** — без какой-либо регистрации:

```python
from spectroview.api import preprocessing

# "asls", "airpls", "modpoly", "snip", ... any pybaselines.Baseline method name
Y_corrected, Y_baseline = preprocessing.subtract_baseline(x, Y, {"mode": "asls", "lam": 1e6})
```

Если в будущей версии pybaselines появится новый метод, он сразу станет доступен в SPECTROview без каких-либо изменений кода с обеих сторон.

---

## Шаблоны моделей аппроксимации и графиков { #fit-model-plot-templates }

И модели аппроксимации, и конфигурации графиков представляют собой обычные JSON-файлы в папке под вашим управлением — SPECTROview (GUI или API) просто сканирует эту папку. Это означает, что наполнять их может любой внешний инструмент, скрипт или репозиторий шаблонов под контролем версий:

```python
from spectroview.api import fitting, graphs, settings

# Fit-model templates live under the SPECTROview Working Folder. Point it at a
# shared location — set_working_folder() creates fit_model/plot_recipe/plot_style/ai_chat_history/template
# subfolders and config.json under it:
settings.set_working_folder("./shared_spectroview")
# ...then any *.json dropped into the fit_model folder (by this API, the GUI, or
# hand-written) is immediately visible:
names = fitting.list_fit_model_templates(settings.get_fit_model_folder())

# Plot recipes work the same way, via a folder-backed store:
graphs.save_plot_recipe("./shared_spectroview/plot_recipe", "QC Dashboard", configs=[...])
```

JSON-структура шаблона модели аппроксимации имеет вид `{"0": {"peak_labels": [...], "peak_models": {...}, "baseline": {...}, "range_min": ..., "range_max": ..., "fit_params": {...}}}` — вы можете записывать такой JSON напрямую, если генерируете шаблоны во внешней системе, либо создать шаблон через `fitting.build_fit_model()` / `fitting.save_fit_model_template()` и изучить полученный результат.

---

## Пользовательские формы пиков { #custom-peak-shapes }

Реестр форм пиков для пакетной аппроксимации (`spectroview.fit_engine.models.BATCHED_MODELS`) представляет собой обычный словарь Python, сопоставляющий имени формы кортеж `(eval_fn, jacobian_fn, param_names)`. Он не относится к публичному API (стабильность между релизами не гарантируется), но если вам нужна форма пика, которой нет в стандартной поставке SPECTROview, её следует добавить именно в этот реестр — а также в `spectroview.fit_engine.scalar_models.PEAK_MODEL_REGISTRY` для резервного непакетного пути. Это требует изменения исходного кода, а не расширения во время выполнения: в настоящее время зарегистрировать новую форму пика из скрипта без модификации установленного пакета невозможно.
