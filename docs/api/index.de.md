# API { #api }

Die Python-API von SPECTROview ermöglicht es Ihnen, Arbeitsabläufe zu automatisieren und programmgesteuert mit den zentralen spektroskopischen Verarbeitungs-Engines von SPECTROview zu interagieren. Sie ist vollständig von der Qt-GUI entkoppelt (keine `QApplication` erforderlich) und eignet sich daher ideal für Jupyter-Notebooks, Batch-Skripte und CI-Pipelines.

Alle Module werden über das Paket `spectroview.api` bereitgestellt.

## API-Module { #api-modules }

- **[`spectroview.api.workspace`](spectra.md#stateful-workspaces)** — `SpectraWorkspace` / `MapsWorkspace`: zustandsbehaftete Sitzungen, die Laden, Vorverarbeiten, Anpassen, Sammeln von Ergebnissen sowie Speichern und Laden an einem Ort vereinen. Die von ihnen geschriebenen und gelesenen Dateien verwenden *genau dasselbe Format* wie die Funktion **Arbeit speichern** (Save work) der GUI — ein vollständig per Skript erstellter Arbeitsbereich lässt sich direkt in der GUI öffnen und umgekehrt.
- **[`spectroview.api.io`](spectra.md#loading-data)** — Dateileser (WDF, SPC, L6S, L6M, NGS, NGC, CSV, TXT, DAT), Datensatzimport, Ergebnisexport sowie ein Konverter für Renishaw-Karten ohne grafische Oberfläche.
- **[`spectroview.api.preprocessing`](spectra.md#preprocessing)** — Zustandslose Operationen auf Array-Ebene: Basislinienabzug, Zuschneiden und Normierung.
- **[`spectroview.api.fitting`](spectra.md#batch-fitting)** — Schnittstelle zur vektorisierten Batch-Fit-Engine (VBF): Modellerstellung, Batch-Fitting und CRUD-Verwaltung von Fit-Modellvorlagen.
- **[`spectroview.api.analysis`](2dmap.md#multivariate-analysis-mva)** — Multivariate Analyse: PCA und NMF.
- **[`spectroview.api.graphs`](graphs.md)** — Plots in Publikationsqualität, die pixelgenau der Darstellung in der GUI entsprechen, für alle 9 unterstützten Plot-Stile sowie CRUD-Verwaltung von Plot-Vorlagen.
- **[`spectroview.api.calculators`](calculators.md)** — Physikalische Rechner für Optik und Einheitenumrechnungen.
- **`spectroview.api.settings`** — Lese- und Schreibzugriff ohne grafische Oberfläche auf die persistente Konfiguration (Standardwerte für Fits, Modellordner, zuletzt verwendetes Verzeichnis, Oberflächensprache über `get_language()` / `set_language()`).
- **`spectroview.api.exceptions`** — Die im gesamten Paket ausgelöste `SpectroviewError`-Hierarchie (`LoadError`, `FitModelError`, `FitError`, `WorkspaceError`, `TemplateError`).

## Die API erkunden { #exploring-the-api }

- [Arbeitsbereich Spektren](spectra.md): Diskrete Spektren laden, vorverarbeiten und die Batch-Fit-Engine ausführen — sowohl zustandsbehaftet über `SpectraWorkspace` als auch direkt auf Array-Ebene.
- [Arbeitsbereich Karten & 2D-Analyse](2dmap.md): Hyperspektral- und Wafer-Karten laden, jedes Pixel anpassen, Heatmaps sowie Linienprofile erstellen und PCA/NMF durchführen.
- [Datenvisualisierung (Diagramme)](graphs.md): Native SPECTROview-Plots für alle Plot-Stile per Skript reproduzieren sowie wiederverwendbare Plot-Vorlagen nutzen.
- [Schnellrechner](calculators.md): Programmgesteuerte Umrechner für optische und spektroskopische Einheiten.
- [SPECTROview erweitern](extending.md): Die tatsächlichen dateibasierten Erweiterungspunkte (benutzerdefinierte Basislinienmethoden, Ordner für Fit-Modelle und Plot-Vorlagen) — ein Plugin-Framework gibt es nicht; stattdessen erläutert diese Seite die verfügbaren Möglichkeiten.

## Fehlerbehandlung { #error-handling }

Jede Funktion in `spectroview.api` löst im Fehlerfall eine Unterklasse von `spectroview.api.exceptions.SpectroviewError` aus — niemals ein rohes `KeyError` oder `ValueError` aus der zugrunde liegenden Modellschicht und niemals einen Qt-Dialog (der ein Skript ohne grafische Oberfläche blockieren würde). Fangen Sie `SpectroviewError` allgemein ab oder gezielt einen spezifischen Untertyp (`LoadError`, `FitModelError`, `FitError`, `WorkspaceError`, `TemplateError`):

```python
from spectroview.api import io
from spectroview.api.exceptions import LoadError

try:
    data = io.load_spectra("missing_file.wdf")
except LoadError as e:
    print(f"Could not load spectra: {e}")
```
