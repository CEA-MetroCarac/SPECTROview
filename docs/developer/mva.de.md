# **Multivariate Analyse (MVA) Raman-spektroskopischer Daten** { #multivariate-analysis-mva-of-raman-spectroscopic-data }

Dieses Dokument bietet einen technischen Überblick über die in `SPECTROview` implementierten Funktionen zur multivariaten Analyse (MVA).

## **Architektur** { #architecture }

Die MVA-Funktion folgt dem Architekturmuster MVVM (Model-View-ViewModel):

- **Model** (`spectroview/model/analysis/m_mva.py`): Enthält die mathematische Kernlogik der MVA-Verfahren, insbesondere die Hauptkomponentenanalyse (PCA) und die nicht-negative Matrixfaktorisierung (NMF). Verwendet standardmäßige numerische Bibliotheken (`numpy`, `scipy`) ohne zusätzliche Abhängigkeiten wie `scikit-learn`.
- **ViewModel** (`spectroview/viewmodel/analysis/vm_mva.py`): Orchestriert den Datenfluss. Es extrahiert vorverarbeitete Daten aus den aktiven Spektren, ruft das Model zur Berechnung auf, speichert Ergebnisse zwischen und berechnet Rekonstruktionsfehler pro Spektrum.
- **View** (`spectroview/view/components/lower_tabs/`: `v_mva.py`, `v_mva_plots.py`): Stellt die Benutzeroberfläche mit einem Plotbereich aus mehreren Tabs (**Zusammenfassung** (Summary), **Scree-Plot** (Scree Plot), **Ladungen (Loadings)**, **Heatmap**, **Scores** und **Residuen** (Residuals)), interaktiven Parametersteuerelementen sowie `matplotlib`-Navigationssymbolleisten für jeden Plot bereit.

## **Algorithmische Implementierung** { #algorithm-implementation }

Die PCA-Implementierung in `SPECTROview` nutzt die Singulärwertzerlegung (SVD) auf mittelwertzentrierten Daten.

### **Visualisierungserweiterungen** { #visualization-enhancements }

`SPECTROview` verfügt über eine umfassende Tab-Oberfläche zur Visualisierung der Ergebnisse:

- **Zusammenfassung**: 2×2-Dashboard mit Scree-Plot (nur PCA), den ersten 3 Ladungen und den primären Score-Streudiagramm-Projektionen
- **Scree-Plot** (nur PCA): Balkendiagramm der Einzelvarianz + kumulative Kurve
- **Ladungen**: Überlagerte spektrale Ladungsprofile mit Flächenfüllung unter der Kurve zur besseren Übersichtlichkeit
- **Heatmap**: Räumliche 2D-Karte der Scores der ausgewählten Komponente, sofern die Spektrennamen `(x, y)`-Koordinaten enthalten
- **Scores**: 2D-Streudiagramm mit interaktiver Achsenauswahl
- **Residuen**: Rekonstruktionsfehler pro Spektrum zur Beurteilung der Modellqualität

## **Implementierte Methoden** { #implemented-methods }

### **1. Hauptkomponentenanalyse (PCA)** { #1-principal-component-analysis-pca }

Die PCA ist ein Verfahren zur Dimensionsreduktion, das Spektren so in ein neues Koordinatensystem transformiert, dass die größte Varianz jeder Projektion der Daten auf der ersten Koordinate (der ersten Hauptkomponente) liegt.

- **Implementierungsdetails**: Implementiert mittels Singulärwertzerlegung (`numpy.linalg.svd`) auf optional mittelwertzentrierten Daten.
- **Parameter**:
  - **Komponenten** (Components): Anzahl der beizubehaltenden Hauptkomponenten (2–50).
  - **Mittelwertzentrierung** (Mean centering): Legt fest, ob das mittlere Spektrum vor der SVD subtrahiert wird (Standard: aktiviert). Dies ist gängige Praxis bei der spektralen PCA.
- **Visualisierungen**:
  - *Zusammenfassung*: 2×2-Übersichts-Dashboard, das Scree-Plot, Ladungen und primäre Score-Paare kombiniert.
  - *Scree-Plot*: Zeigt den Prozentsatz der durch jede Hauptkomponente erklärten Varianz sowie die kumulative Varianz an.
  - *Ladungen*: Visualisiert den spektralen Beitrag zu jeder Hauptkomponente, mit dem Prozentsatz der erklärten Varianz in der Legende.
  - *Heatmap*: Räumlicher 2D-Score-Plot für eine auswählbare Komponente, wenn `(x, y)`-Koordinaten in den Spektrennamen kodiert sind.
  - *Scores*: Stellt ein 2D-Streudiagramm der auf die Hauptkomponenten projizierten Spektren mit interaktiver Achsenauswahl bereit.
  - *Residuen*: L2-Rekonstruktionsfehler pro Spektrum zur Beurteilung der Modellqualität.

### **2. Nicht-negative Matrixfaktorisierung (NMF)** { #2-non-negative-matrix-factorization-nmf }

Die NMF faktorisiert die Datenmatrix in zwei nicht-negative Matrizen (üblicherweise W und H). Dies ist für spektroskopische Daten besonders nützlich, da es additive, nicht-negative Komponenten liefert, die häufig physikalischen Reinkomponenten (*Endmembers*) entsprechen.

- **Implementierungsdetails**: Verwendet die multiplikativen Aktualisierungsregeln nach Lee & Seung. Dabei werden die Matrizen iterativ aktualisiert, um die Frobenius-Norm des Rekonstruktionsfehlers zu minimieren.
- **Parameter**:
  - **Komponenten**: Anzahl der aus dem Datensatz zu extrahierenden nicht-negativen Komponenten (bzw. Endmembers).
  - **Max. Iterationen** (Max Iterations): Maximale Anzahl multiplikativer Aktualisierungsiterationen, die der Algorithmus ausführen darf. Konvergiert der Algorithmus nicht innerhalb dieses Limits, wird er beendet.
  - **Toleranz** (Tolerance): Die Konvergenzschwelle. Der Algorithmus bricht vorzeitig ab, wenn die relative Änderung des Rekonstruktionsfehlers zwischen aufeinanderfolgenden Iterationen unter diesen Wert fällt.
  - **Zufalls-Seed** (Random Seed): Startwert für den Zufallszahlengenerator zur Initialisierung der Matrizen. Ein fester Seed stellt sicher, dass die NMF-Ergebnisse über verschiedene Durchläufe hinweg reproduzierbar sind (Standard: 42).
- **Visualisierungen**:
  - *Zusammenfassung*: Übersichts-Dashboard der Ladungen und der primären NMF-Score-Projektionen.
  - *Ladungen*: Zeigt die nicht-negativen spektralen Komponenten (Endmembers) an.
  - *Heatmap*: Räumliche 2D-Häufigkeitskarte für eine auswählbare NMF-Komponente, sofern `(x, y)`-Koordinaten in den Spektrennamen vorhanden sind.
  - *Scores*: Bietet ein 2D-Streudiagramm der Konzentrationen bzw. Anteile dieser Komponenten im gesamten Datensatz.
  - *Residuen*: L2-Rekonstruktionsfehler pro Spektrum.

> **Warum besitzt NMF keinen Scree-Plot?**  
> Im Gegensatz zur PCA, die orthogonale Komponenten strikt nach der maximal erklärten Varianz ordnet, ermittelt die NMF nicht-orthogonale, additive Komponenten (Endmembers), die sich überlagern. Da NMF-Komponenten weder orthogonal sind noch sequenziell nach Varianz geordnet werden, lässt sich für jede einzelne Komponente kein isolierter Prozentsatz der „erklärten Varianz“ berechnen. Zur Bewertung eines NMF-Modells sollten Sie den Plot **Residuen** zur Beurteilung der Rekonstruktionsqualität sowie die **Ladungen** heranziehen, um sicherzustellen, dass die Komponenten physikalisch sinnvollen Spektren entsprechen.

## **Benutzeroberfläche** { #user-interface }

Die MVA-Ansicht verwendet ein zweigeteiltes Panel-Layout:

### **Linkes Panel (Steuerelemente)** { #left-panel-controls }

- **Methodenauswahl**: Optionsfelder (Radio-Buttons) zum Umschalten zwischen `PCA` und `NMF`.
- **Parametergruppen**: Separate Gruppenfelder für PCA- und NMF-Parameter, die jeweils nur die für die gewählte Methode relevanten Parameter anzeigen.
- **Schaltfläche „Analyse starten“** (Run Analysis): Führt die Analyse aus.
- **Exportbereich**: Sendet die Scores an den Arbeitsbereich **Diagramme** (`Graphs`).

### **Rechtes Panel (Plot-Tabs)** { #right-panel-tabbed-plots }

Jeder Plot-Typ belegt einen eigenen Tab in einem `QTabWidget`:

- **Zusammenfassung**: Kombiniertes Übersichtsraster (Scree-Plot [PCA], erste 3 Ladungen und primäre Score-Streudiagramm-Projektionen).
- **Scree-Plot** (nur PCA): Erklärte Varianz pro Komponente.
- **Ladungen**: Spektrale Ladungsprofile.
- **Heatmap**: Räumliche 2D-Karte der Scores der gewählten Komponente (erfordert `(x, y)`-Koordinaten in den Spektrennamen).
- **Scores**: Interaktives 2D-Streudiagramm mit Achsenauswahl.
- **Residuen**: Rekonstruktionsqualität pro Spektrum.

Jeder Tab enthält eine `matplotlib`-`NavigationToolbar` zum Zoomen, Verschieben und Speichern.

## **Datenpipeline** { #data-pipeline }

1. **Auswahl**: Die MVA arbeitet ausschließlich mit den *aktiven* (angehakten) Spektren im Arbeitsbereich von `SPECTROview`.
2. **Vorverarbeitung**: `VMMVA` bezieht die Daten direkt aus `SpectraStore` (`md.Y` bei vorverarbeiteten Daten, mit Rückfall auf `md.Y0` für Rohdaten) und arbeitet somit auf Daten, die bereits vom Benutzer konfigurierte Vorverarbeitungsschritte (`baseline`-Subtraktion, Zuschneiden, X-Korrektur) durchlaufen haben.
3. **Interpolation**: Weisen die ausgewählten Spektren unterschiedliche Wellenzahlachsen auf, werden sie automatisch auf ein gemeinsames Raster (das Raster des ersten aktiven Spektrums) interpoliert, um eine konsistente Datenmatrix aufzubauen.
4. **Export**: Die Ergebnisse (Scores bei PCA, W-Matrix bei NMF) können zur weiteren Visualisierung und zum Vergleich direkt in den Arbeitsbereich **Diagramme** exportiert werden.

## **Zitierung und Referenzen** { #citations-and-references }

Wenn Sie diese Funktionen nutzen, um Ergebnisse für eine Publikation zu erzeugen, bitten wir Sie, die entsprechende Fachliteratur zu den implementierten Algorithmen zu zitieren:

[1] **PCA-Algorithmus**: Wold, S., Esbensen, K., and Geladi, P., "Principal component analysis." *Chemometrics and intelligent laboratory systems* 2.1-3 (1987): 37-52.

[2] **NMF-Algorithmus**: Lee, Daniel D., and H. Sebastian Seung, "Learning the parts of objects by non-negative matrix factorization." *Nature* 401.6755 (1999): 788-791.
