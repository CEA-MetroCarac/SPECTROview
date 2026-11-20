# Espace Spectra

Cette section explique comment manipuler des spectres 1D discrets : chargement, prétraitement et ajustement à l'aide du moteur Vectorized Batch Fit (VBF) — aussi bien via la session à état persistant `SpectraWorkspace` (recommandée) que via les fonctions bas niveau sur tableaux sur lesquelles elle repose.

---

## Workspaces à état persistant

`spectroview.api.workspace.SpectraWorkspace` reflète l'espace Spectra de l'interface graphique : il contient vos spectres, suit l'état du prétraitement et du modèle d'ajustement pour chaque spectre, et lit/écrit exactement le même format de fichier `.spectra` que le bouton « Save work » de l'interface. Un espace construit entièrement par script s'ouvre directement dans l'interface graphique, et un fichier `.spectra` enregistré depuis l'application se charge directement ici.

```python
from spectroview.api import workspace, fitting

ws = workspace.SpectraWorkspace()
ws.load_files(["sample_a.wdf", "sample_b.wdf", "sample_c.txt"])
print(ws)  # SpectraWorkspace(3 spectra: ['sample_a', 'sample_b', 'sample_c'])

# Prétraiter tous les spectres de l'espace de travail
ws.crop(range_min=400.0, range_max=800.0)
ws.set_baseline({"mode": "arpls", "smoothing_factor": 1e5})
ws.subtract_baseline()

# Associer un modèle d'ajustement et lancer l'optimisation
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

df = ws.collect_results()   # DataFrame pandas, une ligne par spectre
print(df)

# Sauvegarder — ce fichier s'ouvre directement dans l'interface SPECTROview
ws.save("session.spectra")
```

Pour recharger une session ultérieurement (créée via cette API ou enregistrée depuis l'interface graphique) :

```python
ws2 = workspace.SpectraWorkspace.load("session.spectra")
```

Chaque méthode de prétraitement ou d'ajustement accepte un argument optionnel `names=[...]` pour cibler un sous-ensemble de spectres au lieu de tous les traiter, par exemple `ws.crop(range_min=400, range_max=800, names=["sample_a"])`.

!!! note "Fichiers historiques"
    Les très anciens fichiers `.spectra` (format v1, antérieur au format d'archive ZIP actuel) ne sont pris en charge que par l'interface graphique. `SpectraWorkspace.load()` lève une exception `WorkspaceError` sur ces fichiers — ouvrez le fichier une fois dans l'application puis enregistrez-le de nouveau pour le convertir au format actuel.

---

## Chargement des données

Le module `spectroview.api.io` fournit des fonctions unifiées pour charger des spectres depuis de nombreux formats (WDF, SPC, L6S, NGS, CSV, TXT, DAT) — c'est ce qu'utilise en interne `SpectraWorkspace.load_files()`, et cela reste également très pratique pour les traitements directs sur tableaux présentés ci-dessous.

```python
from spectroview.api import io
import numpy as np

spectra_dict = io.load_spectra("my_sample_spectra.wdf")

# Chaque élément de la liste est un spectre (les séries WDF peuvent en contenir plusieurs)
items = spectra_dict["items"]
print(f"Number of spectra loaded: {len(items)}")

# Chaque élément contient : "name", "x0" (axe des nombres d'onde), "y0" (intensités), "metadata"
first = items[0]
print("Wavenumber axis (x0):", first["x0"].shape)   # shape (M,)
print("Intensities (y0):    ", first["y0"].shape)   # shape (M,)
print("Laser Wavelength:", first["metadata"].get("Laser Wavelength (nm)"))
```

### Charger plusieurs fichiers en un lot

Pour manipuler un ensemble de spectres sous forme d'une matrice 2D unique (plutôt qu'un `SpectraWorkspace`), utilisez `io.load_spectra_to_matrix()` :

```python
import glob

file_paths = sorted(glob.glob("./measurements/*.wdf"))
data = io.load_spectra_to_matrix(file_paths)

x     = data["x"]      # float64[M]   — axe des nombres d'onde partagé
Y     = data["Y"]      # float64[N,M] — matrice d'intensités (N spectres)
names = data["names"]  # list[str]    — noms des spectres
```

!!! note
    `load_spectra_to_matrix()` interpole tous les spectres sur l'axe X du premier fichier si leurs grilles spectrales diffèrent légèrement.

---

## Prétraitement

Pour un contrôle direct au niveau des tableaux sans `SpectraWorkspace`, `spectroview.api.preprocessing` expose les mêmes opérations de recadrage, ligne de base et normalisation sous forme de fonctions sans état sur les tableaux `(x, Y)`.

```python
from spectroview.api import preprocessing

# Recadrer sur la région d'intérêt
x_crop, Y_crop = preprocessing.crop_spectra(x, Y, range_min=400.0, range_max=800.0)

# Soustraction de ligne de base (mêmes algorithmes que l'interface : arpls, airpls, asls, Linear, Polynomial)
baseline_config = {"mode": "arpls", "smoothing_factor": 1e5}
Y_corrected, Y_baseline = preprocessing.subtract_baseline(x_crop, Y_crop, baseline_config)

# Normaliser chaque spectre par son propre maximum
Y_norm = preprocessing.normalize_spectra(Y_corrected)
```

---

## Ajustement par lots

La puissance principale de SPECTROview réside dans son moteur Vectorized Batch Fit (VBF), capable d'ajuster des milliers de spectres en une fraction de seconde. `spectroview.api.fitting` l'expose directement pour les scripts manipulant des tableaux (`SpectraWorkspace.fit()` l'utilise en interne, spectre par spectre).

### Étape 1 : Définir ou charger le modèle d'ajustement

**Option A : Charger un modèle exporté depuis l'interface graphique SPECTROview (ou enregistré via cette API)**

```python
from spectroview.api import fitting

fit_model = fitting.load_fit_model_template("my_gui_model.json")
```

**Option B : Construire le modèle avec `fitting.build_fit_model()`**

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

Chaque dictionnaire de paramètre de pic accepte `value`, `min`, `max` et éventuellement `vary` (booléen, `True` par défaut) ou `fix` (son opposé) ainsi que `expr` (une expression textuelle le reliant à un autre paramètre).

### Étape 2 : Exécuter l'ajustement

```python
fit_params = {"xtol": 1e-4, "ftol": 1e-4, "max_ite": 500}

results = fitting.fit_batch(x_crop, Y_corrected, fit_model, fit_params=fit_params)

print(f"Success rate: {results['success'].mean() * 100:.1f}%")
print(f"Average R-squared: {results['r_squared'].mean():.4f}")

fitted_params_matrix = results["params"]        # shape (N, K)
parameter_names      = results["param_names"]   # ex. ['P1_x0', 'P1_fwhm', ...]
```

Par défaut, `fit_batch` calcule les poids d'ajustement à partir de `fit_params` exactement comme le fait l'interface avant chaque calcul — les points d'intensité négative sont exclus (sauf si `fit_params["fit_negative"]` vaut `True`), ainsi que les points situés sous le seuil de bruit défini par `fit_params["coef_noise"]`. Passez `auto_weights=False` pour ajuster tous les points avec un poids égal, ou fournissez votre propre tableau `weights`.

Pour une matrice de spectres bruts nécessitant d'abord l'application du recadrage et de la ligne de base contenus dans un modèle d'ajustement, utilisez `fitting.apply_fit_model(x, Y, fit_model)`, qui enchaîne les trois étapes (recadrage, ligne de base, ajustement) en un seul appel et retourne également les tableaux `x`/`Y` effectivement ajustés.

### Étape 3 : Exporter les résultats

```python
import pandas as pd
from spectroview.api import io

df_results = pd.DataFrame(results["params"], columns=results["param_names"])
df_results["R_squared"]   = results["r_squared"]
df_results["Fit_Success"] = results["success"]

io.export_results(df_results, "my_fit_results.csv")
```

### Enregistrer un modèle d'ajustement pour le réutiliser

```python
fitting.save_fit_model_template(fit_model, "my_model.json")

# Lister et recharger ultérieurement les modèles d'un dossier
names = fitting.list_fit_model_templates("./models")
fit_model = fitting.load_fit_model_template("./models/my_model.json")
```
