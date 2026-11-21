## **Unterstützte Datenformate** { #supported-data-formats }

Beispiele für alle unterstützten Datentypen finden sich im Ordner [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) des GitHub-Repositorys. Sie können diese Dateien herunterladen, um sich mit den unterstützten Dateiformaten und Datenstrukturen vertraut zu machen und den Umgang mit **SPECTROview** anhand realer Daten zu üben.

| Format | Dateierweiterung | Loader | Datentyp |
|---|---|---|---|
| Text mit Trennzeichen | `.txt`, `.csv` | Automatische Trennzeichen- und Header-Erkennung | Einzelspektren, 2D-Karten, Wafer- und zeitaufgelöste Karten |
| Renishaw WiRE | `.wdf` | Integrierter Binär-Reader | Einzelspektren, Zeitreihen, Linienprofile und 2D-Karten |
| HORIBA LabSpec 6 | `.l6s`, `.l6m` | Integrierter Binär-Reader | Einzelspektren (`.l6s`) und hyperspektrale 2D-Karten (`.l6m`) |
| HORIBA LabSpec 5 | `.ngs`, `.ngc` | Integrierter Binär-Reader | Einzelspektren (`.ngs`) und komprimierte hyperspektrale 2D-Karten (`.ngc`) |
| SPC | `.spc` | Integrierter Binär-Reader | Einzelspektren und 2D-Karten |
| TRPL | `.dat` | Integrierter Parser | Zeitaufgelöste Photolumineszenz-Abklingkurven |
| Tabellendaten | `.xlsx`, `.xls`, `.csv` | `Pandas` & `Openpyxl` | DataFrames zum Plotten |
| Fit-Modellvorlage | `.json` | **Fit-Modell-Editor** (Fit Model Builder) | Peak-Modelle und Basislinienkonfiguration |

### **1. Formate `.wdf` (Renishaw), `.spc`, `.l6s` / `.l6m` (HORIBA LabSpec 6) und `.ngs` / `.ngc` (HORIBA LabSpec 5)** { #1-wdf-renishaw-spc-l6s-l6m-horiba-labspec-6-and-ngs-ngc-horiba-labspec-5-formats }

**SPECTROview** unterstützt binäre Spektrometerformate nativ, ohne dass externe Herstellersoftware erforderlich ist:

- **`.wdf`** — Daten, die mit der Software WiRE auf Renishaw-Geräten aufgezeichnet wurden (diskrete Spektren, Zeit-/Wiederholbarkeitsreihen, 1D-Linienscan-Profile und hyperspektrale 2D-Karten).
- **`.l6s` und `.l6m`** — Native Binärspektren (`.l6s`) sowie unkomprimierte und komprimierte spektrale 2D-Bilder/-Karten (`.l6m`), die mit der Software HORIBA LabSpec 6 aufgezeichnet wurden.
- **`.ngs` und `.ngc`** — Native Binärspektren (`.ngs`) und komprimierte spektrale 2D-Bilder/-Karten (`.ngc`), die mit der Software HORIBA LabSpec 5 aufgezeichnet wurden.
- **`.spc`** — Daten, die mit den Programmen LabSpec 6 / T64000 / GRAMS auf HORIBA- und Thermo-Geräten aufgezeichnet wurden (diskrete Raman-/PL-Spektren und hyperspektrale 2D-Karten).

Sowohl diskrete Spektren als auch hyperspektrale Datensätze werden beim Öffnen oder Ziehen per Drag-and-Drop in das Hauptfenster automatisch und ohne vorherige Konvertierung dem passenden Arbeitsbereich zugeordnet. **SPECTROview** liest außerdem die **Metadaten** der Messgeräte aus und importiert sie (Anregungswellenlänge, Gitter, Objektiv, Laserleistung, Belichtungszeit, Akkumulationen, Spalt-/Konfokalblendengröße, spektrale Einheiten, räumliche Schrittweiten, Aufnahmezeitstempel sowie weitere Geräteparameter wie Detektormodell, Detektortemperatur, SWIFT-Modus und Signal-Rausch-Verhältnis S/N).

### **2. Spektroskopische Daten (`.txt`, `.csv`)** { #2-spectroscopic-data-txt-csv }

**SPECTROview** unterstützt 1D-spektroskopische Daten in den Formaten TXT und CSV. Die Dateien müssen aus **zwei numerischen Datenspalten** bestehen, die durch Semikolons (`;`), Kommas (`,`), Tabulatoren oder Leerzeichen getrennt sind. Vorangestellte Kommentarzeilen (`#`) und Spaltenüberschriften sind optional und werden automatisch erkannt, ohne dass bei Dateien ohne Header der erste Datenpunkt verloren geht:

- **Spalte 1**: Werte der X-Achse (z. B. Raman-Verschiebung in cm⁻¹, Wellenlänge in nm, 2θ in Grad). Absteigende X-Achsen werden automatisch aufsteigend sortiert.
- **Spalte 2**: Zugehörige Intensitätswerte (a.u.).

### **3. Hyperspektrale Daten (`.txt`, `.csv`)** { #3-hyperspectral-data-txt-csv }

**SPECTROview** unterstützt hyperspektrale Daten (2D-Probentischkarten, Wafer-Karten sowie zeitaufgelöste Multi-Scan-Wafer-Karten) in den Formaten TXT und CSV mit folgendem Aufbau:

| X    | Y    | R1 | R2 | R3 | ... | Rn |
|------|------|----|----|----|-----|----|
| **X1** | **Y1** | i1 | i1 | i1 | ... | i1 |
| **X2** | **Y2** | i2 | i2 | i2 | ... | i2 |
| **...** | **...** | ... | ... | ... | ... | ... |
| **Xn** | **Yn** | in | in | in | ... | in |

- Die **erste Zeile** (R1 → Rn) definiert die für alle Spektren gemeinsamen X-Achsenwerte (z. B. Raman-Verschiebung in cm⁻¹).
- Die **ersten beiden Spalten** definieren die räumlichen X- und Y-Koordinaten jedes Spektrums innerhalb der Karte.
- Die **übrigen Spalten** enthalten die zugehörigen Intensitätswerte für jedes Spektrum (I1 → In) von der zweiten bis zur letzten Zeile.

> **Hinweis**: Native `.wdf`-Kartendateien von Renishaw können ohne Konvertierung direkt in **SPECTROview** geladen werden. Im Langformat als Text exportierte 2D-Karten aus Renishaw WiRE (Spalten `#X`, `#Y`, `#Wave`, `#Intensity`) lassen sich mit `spectroview.api.io.convert_renishaw_map(input_path, output_path)` in das breite `.txt`-Matrixformat umwandeln.

### **4. Datentabellen (Excel oder CSV)** { #4-datasheet-excel-or-csv }

Excel-Dateien (`.xlsx`, `.xls`) mit einem oder mehreren Tabellenblättern sowie CSV-Dateien können zur sofortigen Visualisierung und grafischen Darstellung direkt in den Arbeitsbereich **Diagramme** (Graphs) importiert werden.

### **5. Native **SPECTROview**-Formate** { #5-native-spectroview-formats }

Um Arbeitsabläufe speichern und zu einem späteren Zeitpunkt fortsetzen zu können, verwendet **SPECTROview** spezifische Dateierweiterungen für den jeweils aktiven Arbeitsbereich:

- **Arbeitsbereich Karten** (Maps) → `.maps`
- **Arbeitsbereich Spektren** (Spectra) → `.spectra`
- **Arbeitsbereich Diagramme** → `.graphs`
- **Fit-Modellvorlagen** → `.json` (per Drag-and-Drop oder über **Laden** (Load) in den Arbeitsbereichen **Spektren** und **Karten**)
