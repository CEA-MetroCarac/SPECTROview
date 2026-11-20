# Espace Maps et analyse 2D

Cette section montre comment traiter des données de cartographie hyperspectrale (telles que des cartographies Raman de plaquettes *wafer*) : chargement, ajustement de chaque pixel, génération de cartes thermiques et de profils linéaires, et exécution d'analyses multivariées — via la session à état persistant `MapsWorkspace` (recommandée) et, pour la MVA, via les fonctions bas niveau sur tableaux.

---

## Workspaces à état persistant

`spectroview.api.workspace.MapsWorkspace` reproduit l'espace Maps de l'interface graphique : une carte par fichier chargé, dont tous les pixels partagent un modèle d'ajustement, avec des identifiants de pixels nommés `f"{map_name}_({x}, {y})"` (ou `f"{map_name}_({x}, {y})_{k}"` lorsque des cartes résolues en temps contiennent plusieurs balayages aux mêmes coordonnées `(x, y)`), exactement comme dans l'interface. Il lit et écrit le même format de fichier `.maps` que le bouton « Save work » de l'application.

```python
from spectroview.api import workspace, fitting

ws = workspace.MapsWorkspace()          # ou MapsWorkspace(map_type="wafer_300mm") pour les cartes wafer
[map_name] = ws.load_files(["2Dmap_Si.txt"])

fit_model = fitting.load_fit_model_template("fit_model_Si.json")
ws.set_fit_model(fit_model, names=[map_name])
ws.fit(map_names=[map_name])            # un seul appel vectorisé pour tous les pixels de la carte

df = ws.collect_results()               # une ligne par pixel, incluant les coordonnées X/Y
print(df.head())

# Construire la carte thermique d'un paramètre ajusté
xi, yi, zi = ws.get_heatmap(map_name, "ampli_Si")

import matplotlib.pyplot as plt
plt.imshow(zi, extent=[xi.min(), xi.max(), yi.min(), yi.max()], origin="lower", cmap="jet")
plt.colorbar(label="Amplitude")
plt.savefig("heatmap.png", dpi=200)

# Extraire un profil linéaire entre deux points de la carte thermique
profile = ws.extract_profile(map_name, "ampli_Si", point1=(xi.min(), yi.min()), point2=(xi.max(), yi.max()))
print(profile)  # colonnes : X, Y, distance, values

# Sauvegarder — ce fichier s'ouvre directement dans l'interface SPECTROview
ws.save("session.maps")
```

`get_heatmap(map_name, value_col, ...)` accepte `value_col="Intensity"` ou `"Area"` (calculés directement à partir des intensités brutes, sans ajustement préalable) ou n'importe quelle colonne de paramètre d'ajustement présente dans `df_fit_results` (appelez d'abord `collect_results()`). Pour les cartes de type *wafer* (`map_type != "2Dmap"`), la grille est construite par interpolation de points dispersés sur le rayon de plaquette défini par `map_type` ; pour `"2Dmap"`, il s'agit d'un pivot exact rapide sur la grille régulière de l'acquisition.

Recharger une session ultérieurement (créée via cette API ou enregistrée depuis l'interface graphique) :

```python
ws2 = workspace.MapsWorkspace.load("session.maps")
```

`MapsWorkspace` hérite de toutes les méthodes de prétraitement de `SpectraWorkspace` (`crop`, `set_baseline`, `subtract_baseline`, `normalize`, `reinit`) — chacune s'applique simultanément à tous les pixels de la ou des cartes ciblées.

---

## Chargement direct des données de carte

Pour un contrôle direct sur tableaux sans `MapsWorkspace`, `spectroview.api.io.load_map()` renvoie le DataFrame brut de la carte (utilisé en interne par `MapsWorkspace.load_files()`) :

```python
from spectroview.api import io
import numpy as np

# Les fichiers WDF, SPC, L6M et NGC retournent (DataFrame, metadata_dict) ; TXT/CSV retournent uniquement un DataFrame.
result = io.load_map("my_wafer_map.wdf")
df_map, metadata = result if isinstance(result, tuple) else (result, {})
```

Les deux premières colonnes du DataFrame retourné contiennent toujours les coordonnées spatiales (`X`, `Y`) ; toutes les colonnes suivantes correspondent aux valeurs de nombre d'onde (sous forme de chaînes de caractères) :

```python
x_coords = df_map["X"].to_numpy(dtype=float)
y_coords = df_map["Y"].to_numpy(dtype=float)

wn_cols  = [c for c in df_map.columns if c not in ("X", "Y")]
x_axis   = np.array([float(c) for c in wn_cols])
Y_matrix = df_map[wn_cols].to_numpy(dtype=float)
```

`Y_matrix` peut être passé directement à `spectroview.api.fitting.fit_batch()` — voir [Espace Spectra : Ajustement par lots](spectra.md#ajustement-par-lots).

---

## Analyse multivariée (MVA)

L'analyse multivariée explore de grands jeux de données complexes sans nécessiter de connaissance préalable des positions de pics. `spectroview.api.analysis` fournit l'ACP (PCA) et la NMF, applicables à toute matrice `(n_spectra, n_wavenumbers)` — par exemple `Y_matrix` ci-dessus, ou `Y0` extrait du magasin sous-jacent d'un `MapsWorkspace`.

### Analyse en Composantes Principales (PCA)

```python
from spectroview.api import analysis, preprocessing

Y_norm = preprocessing.normalize_spectra(Y_matrix)
pca_result = analysis.pca(Y_norm, n_components=3, center=True)

print("Explained Variance Ratios:", pca_result.explained_variance_ratio)
scores   = pca_result.scores     # (n_spectra, n_components)
loadings = pca_result.loadings   # (n_components, n_wavenumbers)
```

### Factorisation par Matrices Non Négatives (NMF)

La NMF contraint les composantes et les scores à rester strictement positifs, fournissant souvent des spectres de « composantes pures » plus directement interprétables physiquement que l'ACP.

```python
nmf_result = analysis.nmf(Y_norm, n_components=3, max_iter=1000)
scores_w   = nmf_result.W
loadings_h = nmf_result.H
```

### Erreur de reconstruction

```python
errors = analysis.reconstruction_error(Y_norm, pca_result.scores, pca_result.loadings, pca_result.mean_spectrum)
# errors : (n_spectra,) résidu L2 par spectre — utile pour repérer les spectres atypiques
```

### Export des résultats MVA

```python
import pandas as pd

df_pca = pd.DataFrame({"X": x_coords, "Y": y_coords})
for i in range(pca_result.n_components):
    df_pca[f"PCA_Score_PC{i+1}"] = scores[:, i]

df_pca.to_csv("map_with_pca_scores.csv", index=False)
```

Pour visualiser `df_pca` sous forme de carte thermique exactement comme `MapsWorkspace.get_heatmap()`, utilisez `spectroview.model.spectra.heatmap.build_heatmap_grid(x_coords, y_coords, scores[:, 0], map_type="2Dmap")`.
