# Arbeitsbereich Spektren { #spectra-workspace }

Dieser Abschnitt zeigt die Verarbeitung diskreter 1D-Spektren: das Laden, Vorverarbeiten und Anpassen mit der vektorisierten Batch-Fit-Engine (Vectorized Batch Fit, VBF) — sowohl über die zustandsbehaftete `SpectraWorkspace`-Sitzung (empfohlen) als auch über die arraybasierten Low-Level-Funktionen, auf denen sie aufbaut.

---

## Zustandsbehaftete Arbeitsbereiche { #stateful-workspaces }

`spectroview.api.workspace.SpectraWorkspace` bildet den Arbeitsbereich Spektren der grafischen Benutzeroberfläche direkt ab: Die Klasse verwaltet Ihre Spektren, verfolgt den Zustand von Vorverarbeitung und Fit-Modell für jedes Spektrum und liest bzw. schreibt exakt dasselbe `.spectra`-Dateiformat, das auch die Schaltfläche **Arbeit speichern** (Save work) in der GUI erzeugt. Ein vollständig per Skript erstellter Arbeitsbereich lässt sich direkt in der GUI öffnen, und eine in der GUI gespeicherte `.spectra`-Datei kann unmittelbar hier geladen werden.

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

Um eine Sitzung später erneut zu laden (unabhängig davon, ob sie über diese API erstellt oder aus der GUI gespeichert wurde):

```python
ws2 = workspace.SpectraWorkspace.load("session.spectra")
```

Jede Vorverarbeitungs- und Fit-Methode akzeptiert eine optionale Liste `names=[...]`, um gezielt nur eine Teilmenge der Spektren statt aller Spektren zu bearbeiten, z. B. `ws.crop(range_min=400, range_max=800, names=["sample_a"])`.

!!! note "Ältere Dateien"
    Sehr alte `.spectra`-Dateien (Format v1, noch vor dem aktuellen ZIP-basierten Speicherformat) werden nur von der GUI unterstützt. `SpectraWorkspace.load()` löst bei solchen Dateien einen `WorkspaceError` aus — öffnen Sie die Datei einmalig in der GUI und speichern Sie sie erneut, um sie in das aktuelle Format zu konvertieren.

---

## Daten laden { #loading-data }

Das Modul `spectroview.api.io` bietet einheitliche Funktionen zum Laden von Spektren aus verschiedenen Formaten (WDF, SPC, L6S, NGS, CSV, TXT, DAT) — diese werden intern von `SpectraWorkspace.load_files()` verwendet und eignen sich auch eigenständig für den nachfolgend beschriebenen arraybasierten Low-Level-Workflow.

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

### Mehrere Dateien im Batch laden { #loading-multiple-files-as-a-batch }

Um eine Sammlung von Spektren gemeinsam als eine einzige 2D-Matrix (statt in einem `SpectraWorkspace`) zu verarbeiten, verwenden Sie `io.load_spectra_to_matrix()`:

```python
import glob

file_paths = sorted(glob.glob("./measurements/*.wdf"))
data = io.load_spectra_to_matrix(file_paths)

x     = data["x"]      # float64[M]   — shared wavenumber axis
Y     = data["Y"]      # float64[N,M] — intensity matrix (N spectra)
names = data["names"]  # list[str]    — spectrum names
```

!!! note
    `load_spectra_to_matrix()` interpoliert alle Spektren auf die X-Achse der ersten Datei, falls ihre Achsen leicht voneinander abweichen.

---

## Vorverarbeitung { #preprocessing }

Für die direkte Kontrolle auf Array-Ebene ohne `SpectraWorkspace` stellt `spectroview.api.preprocessing` dieselben Operationen zum Zuschneiden, zur Basislinienkorrektur und zur Normierung als zustandslose Funktionen auf `(x, Y)`-Arrays bereit.

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

## Batch-Fitting { #batch-fitting }

Die zentrale Stärke von SPECTROview liegt in der vektorisierten Batch-Fit-Engine (VBF), mit der sich Tausende Spektren in Sekundenbruchteilen anpassen lassen. `spectroview.api.fitting` macht diese Engine direkt für Workflows auf Array-Ebene zugänglich (ein Aufruf von `SpectraWorkspace.fit()` nutzt sie intern pro Spektrum).

### Schritt 1: Fit-Modell definieren oder laden { #step-1-define-or-load-the-fit-model }

**Option A: Eine aus der SPECTROview-GUI exportierte (oder über diese API gespeicherte) Vorlage laden**

```python
from spectroview.api import fitting

fit_model = fitting.load_fit_model_template("my_gui_model.json")
```

**Option B: Das Modell mit `fitting.build_fit_model()` erstellen**

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

Jedes Parameter-Dictionary eines Peaks akzeptiert `value`, `min`, `max` sowie optional `vary` (boolescher Wert, Standardwert `True`) bzw. `fix` (das Gegenteil davon) und `expr` (einen String-Ausdruck, der den Parameter mit einem anderen verknüpft).

### Schritt 2: Fit ausführen { #step-2-execute-the-fit }

```python
fit_params = {"xtol": 1e-4, "ftol": 1e-4, "max_ite": 500}

results = fitting.fit_batch(x_crop, Y_corrected, fit_model, fit_params=fit_params)

print(f"Success rate: {results['success'].mean() * 100:.1f}%")
print(f"Average R-squared: {results['r_squared'].mean():.4f}")

fitted_params_matrix = results["params"]        # shape (N, K)
parameter_names      = results["param_names"]   # e.g. ['P1_x0', 'P1_fwhm', ...]
```

Standardmäßig leitet `fit_batch` die Fit-Gewichte genau so aus `fit_params` ab, wie es die GUI vor jedem Fit tut — Punkte mit negativer Intensität werden ausgeschlossen (sofern `fit_params["fit_negative"]` nicht `True` ist), ebenso Punkte unterhalb einer durch `fit_params["coef_noise"]` festgelegten Rauschschwelle. Übergeben Sie `auto_weights=False`, um jeden Punkt mit gleichem Gewicht anzupassen, oder übergeben Sie Ihr eigenes `weights`-Array.

Für eine Matrix aus Rohspektren, auf die zunächst noch die Schritte zum Zuschneiden und zur Basislinienkorrektur aus einem Fit-Modell angewendet werden müssen, verwenden Sie `fitting.apply_fit_model(x, Y, fit_model)`. Diese Funktion kombiniert alle drei Schritte (Zuschneiden, Basislinie, Fit) in einem einzigen Aufruf und gibt zusätzlich die tatsächlich angepassten `x`/`Y`-Daten zurück.

### Schritt 3: Ergebnisse exportieren { #step-3-export-results }

```python
import pandas as pd
from spectroview.api import io

df_results = pd.DataFrame(results["params"], columns=results["param_names"])
df_results["R_squared"]   = results["r_squared"]
df_results["Fit_Success"] = results["success"]

io.export_results(df_results, "my_fit_results.csv")
```

### Fit-Modell zur Wiederverwendung speichern { #saving-a-fit-model-for-reuse }

```python
fitting.save_fit_model_template(fit_model, "my_model.json")

# List and reload templates from a folder later
names = fitting.list_fit_model_templates("./models")
fit_model = fitting.load_fit_model_template("./models/my_model.json")
```
