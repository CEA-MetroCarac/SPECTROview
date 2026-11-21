# Datenvisualisierung (Diagramme) { #data-visualization-graphs }

Das Modul `spectroview.api.graphs` ermöglicht es Ihnen, programmatisch Diagramme in Publikationsqualität zu erzeugen, die das native Erscheinungsbild des SPECTROview-Arbeitsbereichs **Diagramme** (Graphs) für alle 9 unterstützten Plot-Stile pixelgenau nachbilden. Es delegiert die Darstellung an exakt denselben Rendering-Code, den auch das Diagramme-Widget der GUI verwendet (keine separate Neuimplementierung, keine Näherung), sodass per Skript erzeugte Plots exakt dem entsprechen, was die GUI zeichnet — dieselbe Farb- und Marker-Abfolge, dieselbe Berechnung des 95-%-Konfidenzintervalls (95-%-KI) sowie dasselbe Styling für Boxplots, Punktdiagramme, Trendlinien, Wafer- und 2D-Karten.

---

## Daten vorbereiten { #preparing-data }

Die Plot-Funktionen erwarten einen standardmäßigen Pandas-DataFrame — typischerweise die Ausgabe von `SpectraWorkspace.collect_results()` / `MapsWorkspace.collect_results()` oder über `io.load_dataset()` geladen.

```python
import pandas as pd
from spectroview.api import io

df = io.load_dataset("fit_results.csv")["fit_results"]
print(df.columns)
```

Jede `plot_*`-Funktion akzeptiert ein optionales Argument `ax=` (zeichnet direkt in diese Achse, statt eine neue Abbildung zu erstellen) und gibt das `Axes`-Objekt zurück, in das gezeichnet wurde, sodass Sie eigene Subplots zusammenstellen können.

---

## Kategoriale / XY-Diagramme { #categorical-xy-plots }

### Statistische Punktdiagramme { #statistical-point-plots }

Mittelwert sowie Fehlerbalken für das 95-%-Konfidenzintervall, gruppiert nach einer kategorialen Variable — gut geeignet zum Vergleich von Peak-Parametern über verschiedene Versuchsbedingungen hinweg.

```python
import matplotlib.pyplot as plt
from spectroview.api import graphs

fig, ax = plt.subplots(figsize=(8, 6))
graphs.plot_point(df, x="Sample", y="peak_1_x0", hue="Condition",
                   join=False, dodge=True, title="Peak Shift by Process Condition", ax=ax)
plt.tight_layout()
plt.savefig("point_plot.png", dpi=300)
```

### Liniendiagramme { #line-plots }

Wie `plot_point`, verbindet jedoch die Mittelwerte je X-Wert durch eine Linie mit einem schattierten 95-%-KI-Band — nützlich für Variablen mit einer natürlichen Reihenfolge (z. B. Tiefe, Zeit, Dosis).

```python
graphs.plot_line(df, x="Depth_nm", y="peak_1_ampli", hue="Condition", title="Amplitude vs Depth")
```

### Balkendiagramme { #bar-plots }

```python
graphs.plot_bar(df, x="Sample", y="peak_1_ampli", hue="Condition", show_error_bar=True)
```

### Boxplots { #box-plots }

```python
graphs.plot_box(df, x="Sample", y="peak_1_fwhm", hue="Condition", title="FWHM Distribution Analysis")
```

### Histogramme { #histograms }

```python
graphs.plot_histogram(df, x="peak_1_fwhm", hue="Condition", bins=20, kde=True)
```

### Streudiagramme & Trendlinien { #scatter-plots-trendlines }

```python
graphs.plot_scatter(df, x="peak_1_ampli", y="peak_2_ampli", hue="Condition",
                     title="Peak 1 vs Peak 2 Amplitude Correlation")

graphs.plot_trendline(df, x="peak_1_x0", y="peak_1_fwhm", order=1, hue="Condition",
                       title="Correlation: FWHM vs Peak Position")
```

---

## Räumliche Diagramme { #spatial-plots }

Diese Funktionen erwarten einen strukturierten (*tidy*) DataFrame mit einer Zeile pro (x, y)-Messpunkt und einer numerischen Wertespalte `z` — genau das Format, das `MapsWorkspace.collect_results()` erzeugt.

### 2D-Karten { #2d-maps }

Schnelle, exakte Heatmap auf dem regulären Messgitter der Aufnahme (Raster-/2D-Kartenaufnahmen).

```python
graphs.plot_2dmap(df, x="X", y="Y", z="ampli_Si", cmap="jet", title="Amplitude Map")
```

### Wafer-Karten { #wafer-maps }

Interpolierte Heatmap über einen kreisförmigen Wafer-Umriss, mit optionaler Statistik-Zusammenfassung direkt im Plot.

```python
graphs.plot_wafer(df, x="X", y="Y", z="ampli_Si", wafer_size=300.0, show_stats=True)
```

---

## Plot-Rezepte { #plot-recipes }

Ein Plot-Rezept (*plot recipe*) ist eine benannte, wiederverwendbare Sammlung von Plot-Konfigurationen (jeweils im selben Dictionary-Format, das `MGraph.save()` in der GUI erzeugt), die als JSON-Datei in einem Ordner gespeichert wird — derselbe Mechanismus, den die GUI-Funktion **Plot-Rezept speichern** (Save Plot Recipe) verwendet. (Nicht zu verwechseln mit einer *Stilvorlage*, die ausschließlich das Erscheinungsbild und keine Datenbindungen speichert — siehe `spectroview/model/graphs/m_style_template_store.py`.)

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
