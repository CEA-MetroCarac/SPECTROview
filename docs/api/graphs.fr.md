# Visualisation de données (Graphs)

Le module `spectroview.api.graphs` vous permet de générer par programme des graphiques de qualité publication qui reproduisent au pixel près l'esthétique native de l'espace Graphs de SPECTROview, pour chacun des 9 styles de tracé pris en charge. Il délègue le tracé exactement au même code de rendu que le widget Graphs de l'interface graphique (aucune réimplémentation séparée, aucune approximation), de sorte que les figures produites par script coïncident fidèlement avec celles de l'interface — même cycle de couleurs et de marqueurs, même calcul d'intervalle de confiance à 95 %, mêmes styles de boîtes, points, courbes de tendance, cartes *wafer* et cartes 2D.

---

## Préparer les données

Les fonctions graphiques attendent un DataFrame Pandas standard — typiquement issu de `SpectraWorkspace.collect_results()` / `MapsWorkspace.collect_results()`, ou chargé via `io.load_dataset()`.

```python
import pandas as pd
from spectroview.api import io

df = io.load_dataset("fit_results.csv")["fit_results"]
print(df.columns)
```

Chaque fonction `plot_*` accepte un paramètre optionnel `ax=` (pour tracer sur un axe existant plutôt que de créer une nouvelle figure) et renvoie l'objet `Axes` utilisé, ce qui vous permet de composer vos propres figures multi-panneaux.

---

## Graphiques catégoriels et XY

### Graphiques en points statistiques (`plot_point`)

Moyenne accompagnée de barres d'erreur à intervalle de confiance à 95 %, regroupées par variable catégorielle — idéal pour comparer des paramètres de pics entre conditions expérimentales.

```python
import matplotlib.pyplot as plt
from spectroview.api import graphs

fig, ax = plt.subplots(figsize=(8, 6))
graphs.plot_point(df, x="Sample", y="peak_1_x0", hue="Condition",
                   join=False, dodge=True, title="Peak Shift by Process Condition", ax=ax)
plt.tight_layout()
plt.savefig("point_plot.png", dpi=300)
```

### Courbes (`plot_line`)

Similaire à `plot_point`, mais relie les moyennes par catégorie X par un trait accompagné d'une bande ombrée d'intervalle de confiance à 95 % — adapté aux variables naturellement ordonnées (profondeur, temps, dose, etc.).

```python
graphs.plot_line(df, x="Depth_nm", y="peak_1_ampli", hue="Condition", title="Amplitude vs Depth")
```

### Diagrammes en barres (`plot_bar`)

```python
graphs.plot_bar(df, x="Sample", y="peak_1_ampli", hue="Condition", show_error_bar=True)
```

### Boîtes à moustaches (`plot_box`)

```python
graphs.plot_box(df, x="Sample", y="peak_1_fwhm", hue="Condition", title="FWHM Distribution Analysis")
```

### Histogrammes (`plot_histogram`)

```python
graphs.plot_histogram(df, x="peak_1_fwhm", hue="Condition", bins=20, kde=True)
```

### Nuages de points et courbes de tendance (`plot_scatter` et `plot_trendline`)

```python
graphs.plot_scatter(df, x="peak_1_ampli", y="peak_2_ampli", hue="Condition",
                     title="Peak 1 vs Peak 2 Amplitude Correlation")

graphs.plot_trendline(df, x="peak_1_x0", y="peak_1_fwhm", order=1, hue="Condition",
                       title="Correlation: FWHM vs Peak Position")
```

---

## Graphiques spatiaux

Ces fonctions prennent en entrée un DataFrame structuré avec une ligne par échantillon `(x, y)` et une colonne numérique `z` — exactement le format produit par `MapsWorkspace.collect_results()`.

### Cartes 2D (`plot_2dmap`)

Carte thermique rapide et exacte sur la grille régulière propre à l'acquisition (balayages raster / cartes 2D).

```python
graphs.plot_2dmap(df, x="X", y="Y", z="ampli_Si", cmap="jet", title="Amplitude Map")
```

### Cartes Wafer (`plot_wafer`)

Carte thermique interpolée sur le contour circulaire d'une plaquette (*wafer*), avec annotation optionnelle des statistiques récapitulatives sur la figure.

```python
graphs.plot_wafer(df, x="X", y="Y", z="ampli_Si", wafer_size=300.0, show_stats=True)
```

---

## Recettes de tracé (`Plot Recipes`)

Une recette de tracé (*plot recipe*) est un ensemble nommé et réutilisable de configurations graphiques (chacune structurée comme le dictionnaire généré par `MGraph.save()` dans l'interface) enregistré dans un fichier JSON au sein d'un dossier — exactement le mécanisme utilisé par la fonctionnalité « Save Plot Recipe » de l'interface graphique. (À ne pas confondre avec un *style template*, qui n'enregistre que l'apparence sans liaisons de données — voir `spectroview/model/graphs/m_style_template_store.py`.)

```python
from spectroview.api import graphs

configs = [
    {"plot_style": "scatter", "x": "peak_1_ampli", "y": ["peak_2_ampli"], "z": "Condition"},
    {"plot_style": "box", "x": "Sample", "y": ["peak_1_fwhm"], "z": "Condition"},
]
recipe_id = graphs.save_plot_recipe("./recipes", "Standard QC Plots", configs)

# Lister et recharger ultérieurement
for summary in graphs.list_plot_recipes("./recipes"):
    print(summary["name"], summary["graph_count"])

configs = graphs.load_plot_recipe("./recipes", recipe_id)
graphs.delete_plot_recipe("./recipes", recipe_id)
```
