# Arbeitsbereich Karten & 2D-Analyse { #maps-workspace-2d-analysis }

Dieser Abschnitt zeigt, wie hyperspektrale Kartendaten (z. B. Raman-Wafer-Karten) verarbeitet werden: Laden, Anpassen jedes Pixels, Erstellen von Heatmaps und Linienprofilen sowie Ausführen multivariater Analysen — über die zustandsbehaftete Sitzung `MapsWorkspace` (empfohlen) und, für MVA, über die array-basierten Low-Level-Funktionen.

---

## Zustandsbehaftete Arbeitsbereiche { #stateful-workspaces }

`spectroview.api.workspace.MapsWorkspace` bildet den eigenen Karten-Arbeitsbereich der Benutzeroberfläche nach: eine Karte pro geladener Datei, deren Pixel sich alle ein Fit-Modell teilen, wobei die Pixel-Kennungen exakt wie in der GUI als `f"{map_name}_({x}, {y})"` (bzw. `f"{map_name}_({x}, {y})_{k}"`, wenn zeitaufgelöste Karten wiederholte Messungen an derselben `(x, y)`-Koordinate enthalten) benannt werden. Er liest und schreibt dasselbe `.maps`-Dateiformat, das auch die Funktion **Arbeit speichern** (Save work) der GUI erzeugt.

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

`get_heatmap(map_name, value_col, ...)` akzeptiert `value_col="Intensity"` oder `"Area"` (direkt aus den Rohintensitäten berechnet, ohne dass ein Fit erforderlich ist) sowie jede in `df_fit_results` vorhandene Fit-Parameter-Spalte (rufen Sie dazu zuerst `collect_results()` auf). Bei Wafer-Kartentypen (`map_type != "2Dmap"`) wird das Raster durch Interpolation verstreuter Punkte über den durch `map_type` vorgegebenen Wafer-Radius erzeugt; bei `"2Dmap"` erfolgt eine schnelle, exakte Pivotierung auf dem eigenen regulären Raster der Messung.

Eine Sitzung zu einem späteren Zeitpunkt wieder laden (über diese API erstellt oder aus der GUI gespeichert):

```python
ws2 = workspace.MapsWorkspace.load("session.maps")
```

`MapsWorkspace` erbt alle Vorverarbeitungsmethoden von `SpectraWorkspace` (`crop`, `set_baseline`, `subtract_baseline`, `normalize`, `reinit`) — jede dieser Methoden wird gleichzeitig auf sämtliche Pixel der betreffenden Karte(n) angewendet.

---

## Kartendaten direkt laden { #loading-map-data-directly }

Für eine Kontrolle auf Array-Ebene ohne `MapsWorkspace` gibt `spectroview.api.io.load_map()` den Rohdaten-DataFrame der Karte zurück (dies nutzt `MapsWorkspace.load_files()` intern):

```python
from spectroview.api import io
import numpy as np

# WDF, SPC, L6M, and NGC files return (DataFrame, metadata_dict); TXT/CSV return a DataFrame only.
result = io.load_map("my_wafer_map.wdf")
df_map, metadata = result if isinstance(result, tuple) else (result, {})
```

Die ersten beiden Spalten des zurückgegebenen DataFrames enthalten stets die räumlichen Koordinaten (`X`, `Y`); alle übrigen Spalten entsprechen den Wellenzahlwerten (als Zeichenketten):

```python
x_coords = df_map["X"].to_numpy(dtype=float)
y_coords = df_map["Y"].to_numpy(dtype=float)

wn_cols  = [c for c in df_map.columns if c not in ("X", "Y")]
x_axis   = np.array([float(c) for c in wn_cols])
Y_matrix = df_map[wn_cols].to_numpy(dtype=float)
```

`Y_matrix` kann direkt an `spectroview.api.fitting.fit_batch()` übergeben werden — siehe [Arbeitsbereich Spektren: Batch-Fitting](spectra.md#batch-fitting).

---

## Multivariate Analyse (MVA) { #multivariate-analysis-mva }

Die multivariate Analyse untersucht große, komplexe Datensätze, ohne dass Vorkenntnisse über Peak-Positionen erforderlich sind. `spectroview.api.analysis` stellt PCA und NMF bereit, die auf beliebige `(n_spectra, n_wavenumbers)`-Matrizen angewendet werden können — z. B. auf die obige `Y_matrix` oder auf `Y0` aus dem zugrunde liegenden Datenspeicher eines `MapsWorkspace`.

### Hauptkomponentenanalyse (PCA) { #principal-component-analysis-pca }

```python
from spectroview.api import analysis, preprocessing

Y_norm = preprocessing.normalize_spectra(Y_matrix)
pca_result = analysis.pca(Y_norm, n_components=3, center=True)

print("Explained Variance Ratios:", pca_result.explained_variance_ratio)
scores   = pca_result.scores     # (n_spectra, n_components)
loadings = pca_result.loadings   # (n_components, n_wavenumbers)
```

### Nicht-negative Matrixfaktorisierung (NMF) { #non-negative-matrix-factorization-nmf }

NMF erzwingt streng positive Komponenten und Scores und liefert damit häufig physikalisch besser interpretierbare „Reinkomponenten“-Spektren als die PCA.

```python
nmf_result = analysis.nmf(Y_norm, n_components=3, max_iter=1000)
scores_w   = nmf_result.W
loadings_h = nmf_result.H
```

### Rekonstruktionsfehler { #reconstruction-error }

```python
errors = analysis.reconstruction_error(Y_norm, pca_result.scores, pca_result.loadings, pca_result.mean_spectrum)
# errors: (n_spectra,) per-spectrum L2 residual -- useful for flagging outlier spectra
```

### MVA-Ergebnisse exportieren { #exporting-mva-results }

```python
import pandas as pd

df_pca = pd.DataFrame({"X": x_coords, "Y": y_coords})
for i in range(pca_result.n_components):
    df_pca[f"PCA_Score_PC{i+1}"] = scores[:, i]

df_pca.to_csv("map_with_pca_scores.csv", index=False)
```

Um `df_pca` auf dieselbe Weise als Heatmap zu visualisieren wie `MapsWorkspace.get_heatmap()`, verwenden Sie `spectroview.model.spectra.heatmap.build_heatmap_grid(x_coords, y_coords, scores[:, 0], map_type="2Dmap")`.
