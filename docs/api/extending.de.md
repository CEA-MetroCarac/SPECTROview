# SPECTROview erweitern { #extending-spectroview }

SPECTROview verfügt über kein Plugin- oder Entry-Point-Framework — es existiert keine Registrierung, über die sich eigener Code in die Anwendung einbinden lässt. Stattdessen basiert die Erweiterbarkeit auf zwei dateibasierten Mechanismen, die direkt von der API bereitgestellt werden. Wenn Sie sich fragen: „Wie füge ich SPECTROview ein benutzerdefiniertes X hinzu?“, ist fast immer einer dieser beiden Mechanismen die Antwort.

---

## Benutzerdefinierte Basislinienmethoden { #custom-baseline-methods }

Die Basislinienauswertung (`spectroview.api.preprocessing.subtract_baseline` sowie `set_baseline` in jedem Arbeitsbereich) wählt das Verfahren anhand des String-Namens in `config["mode"]` aus. Über die integrierten Modi `None`, `"Linear"`, `"Polynomial"` und `"arpls"` hinaus **funktioniert jeder `Baseline`-Methodenname aus [pybaselines](https://pybaselines.readthedocs.io/) direkt und ohne Weiteres** — ein Registrierungsschritt ist nicht erforderlich:

```python
from spectroview.api import preprocessing

# "asls", "airpls", "modpoly", "snip", ... any pybaselines.Baseline method name
Y_corrected, Y_baseline = preprocessing.subtract_baseline(x, Y, {"mode": "asls", "lam": 1e6})
```

Fügt pybaselines in einer zukünftigen Version eine neue Methode hinzu, steht sie in SPECTROview sofort und ohne Codeänderungen auf beiden Seiten zur Verfügung.

---

## Fit-Modell- und Plot-Vorlagen { #fit-model-plot-templates }

Sowohl Fit-Modelle als auch Plot-Konfigurationen sind einfache JSON-Dateien in einem von Ihnen verwalteten Ordner — SPECTROview (GUI oder API) scannt lediglich diesen Ordner. Dadurch kann jedes externe Werkzeug, Skript oder versionsverwaltete Vorlagen-Repository sie befüllen:

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

Die JSON-Struktur einer Fit-Modell-Vorlage hat die Form `{"0": {"peak_labels": [...], "peak_models": {...}, "baseline": {...}, "range_min": ..., "range_max": ..., "fit_params": {...}}}` — schreiben Sie eine solche Datei direkt, wenn Sie Vorlagen aus einem externen System generieren, oder erstellen Sie eine Vorlage mit `fitting.build_fit_model()` / `fitting.save_fit_model_template()` und inspizieren Sie die Ausgabe.

---

## Benutzerdefinierte Peak-Formen { #custom-peak-shapes }

Die Batch-Registry für Peak-Formen (`spectroview.fit_engine.models.BATCHED_MODELS`) ist ein einfaches Python-Dictionary, das jeden Formnamen auf `(eval_fn, jacobian_fn, param_names)` abbildet. Sie ist nicht Teil der öffentlichen API (keine Stabilitätsgarantie über Versionen hinweg); falls Sie jedoch eine Peak-Form benötigen, die SPECTROview nicht mitliefert, müsste sie genau dort — sowie in `spectroview.fit_engine.scalar_models.PEAK_MODEL_REGISTRY` für den Fallback-Pfad ohne Batch-Verarbeitung — ergänzt werden. Dies erfordert eine Änderung am Quellcode, keine Laufzeiterweiterung; derzeit gibt es keine Möglichkeit, eine neue Peak-Form aus einem Skript heraus zu registrieren, ohne das installierte Paket zu modifizieren.
