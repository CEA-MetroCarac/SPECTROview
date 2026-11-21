## **Multivariate Analyse (MVA)** { #multivariate-analysis-mva }

**SPECTROview** bietet integrierte Werkzeuge zur multivariaten Analyse (MVA), die bequem über den speziellen **MVA**-Tab sowohl im Arbeitsbereich **Spektren** (`Spectra`) als auch im Arbeitsbereich **Karten** (`Maps`) zugänglich sind.

> **Hinweis**: Für die Ausführung der MVA-Algorithmen sind mindestens 2 aktive Spektren erforderlich.

### **1. Unterstützte Algorithmen** { #1-supported-algorithms }

#### **1.1. Hauptkomponentenanalyse (PCA)** { #11-principal-component-analysis-pca }

Ein statistisches Verfahren, das komplexe spektrale Datensätze orthogonal transformiert und deren Dimensionalität reduziert, um die wichtigsten Varianzmuster sichtbar zu machen. Die Implementierung basiert auf der Singulärwertzerlegung (SVD).

- **Parameter**:
    - **Komponenten** (Components, 2–50): Anzahl der beizubehaltenden Hauptkomponenten.
    - **Mittelwertzentrierung** (Mean centering, ein/aus): Legt fest, ob das mittlere Spektrum vor der Analyse subtrahiert wird. Standard: ein.
- **Verfügbare Visualisierungen**: Übersichts-Dashboard, Scree-Plots (erklärte Varianz), Ladungsprofile der Hauptkomponenten, räumliche 2D-Komponenten-Heatmap, Score-Plots sowie Plots des Rekonstruktionsfehlers (Residuen).

#### **1.2. Nicht-negative Matrixfaktorisierung (NMF)** { #12-non-negative-matrix-factorization-nmf }

Ein leistungsfähiger Zerlegungsalgorithmus, der die Spektraldaten in einzelne, additive, nicht-negative Komponenten zerlegt, die häufig reinen chemischen Komponenten (*Endmembers*) entsprechen.

- **Parameter**:
    - **Komponenten** (2–50): Anzahl der zu extrahierenden nicht-negativen Komponenten.
    - **Max. Iterationen** (Max Iterations, 50–5000): Maximale Anzahl der Aktualisierungsiterationen.
    - **Toleranz** (Tolerance): Konvergenzschwelle für den vorzeitigen Abbruch.
    - **Zufalls-Seed** (Random Seed): Startwert für reproduzierbare Ergebnisse.
- **Verfügbare Visualisierungen**: Übersichts-Dashboard, Komponenten-Ladungen, räumliche 2D-Komponenten-Heatmap, Score-Plots sowie Plots des Rekonstruktionsfehlers (Residuen).

> **Warum gibt es bei der NMF keinen Scree-Plot?**  
> Im Gegensatz zur PCA, die orthogonale Komponenten streng nach dem Maximum der von ihnen erklärten Varianz ordnet, ermittelt die NMF nicht-orthogonale, additive Komponenten (*Endmembers*), die sich überlagern. Da NMF-Komponenten weder orthogonal noch sequenziell nach der Varianz geordnet sind, lässt sich für jede einzelne Komponente kein isolierter Prozentsatz an „erklärter Varianz“ berechnen. Nutzen Sie zur Bewertung eines NMF-Modells den Plot **Residuen** (Residuals), um die Rekonstruktionsqualität zu prüfen, sowie die **Ladungen (Loadings)**, um sicherzustellen, dass die Komponenten physikalisch sinnvollen Spektren entsprechen.

------

### **2. Benutzeroberfläche** { #2-user-interface }

Der MVA-Tab verfügt über ein **zweigeteiltes Layout**:

#### **2.1. Linkes Panel – Steuerelemente** { #21-left-panel-controls }

- **Methodenauswahl**: Wählen Sie über die Optionsfelder zwischen `PCA` und `NMF`.
- **Parameter-Steuerelemente**: Alle anpassbaren Parameter stehen als interaktive Bedienelemente der Benutzeroberfläche (Zahlenfelder, Kontrollkästchen) zur Verfügung. Es werden jeweils nur die für die gewählte Methode relevanten Parameter angezeigt.
- **Start-Schaltfläche**: Klicken Sie auf ▶ **Analyse starten** (Run Analysis), um die Analyse mit den aktuellen Parametern auszuführen.
- **Export**: Senden Sie die Ergebnisse (Scores) zur weiteren Auswertung an den Arbeitsbereich **Diagramme** (`Graphs`).

#### **2.2. Rechtes Panel – Diagramm-Tabs** { #22-right-panel-tabbed-plots }

Die Ergebnisse werden in mehreren Tabs dargestellt, die jeweils über eine eigene interaktive `matplotlib`-Symbolleiste (Zoomen, Verschieben, Speichern) verfügen:

| Tab | PCA | NMF | Beschreibung |
|-----|-----|-----|--------------|
| **Zusammenfassung** (Summary) | ✅ | ✅ | Kombiniertes Übersichts-Dashboard, das Scree-Plot, Ladungen und Scores gleichzeitig darstellt |
| **Scree-Plot** (Scree Plot) | ✅ | - | Balkendiagramm der erklärten Varianz pro Komponente mit kumulativer Kurve |
| **Ladungen (Loadings)** | ✅ | ✅ | Spektrale Ladungsprofile überlagert auf der Wellenzahlachse |
| **Heatmap** | ✅ | ✅ | Räumliche 2D-Score-Karte für eine wählbare Komponente *(sofern räumliche `(x, y)`-Koordinaten in den Spektrennamen vorhanden sind)* |
| **Scores** | ✅ | ✅ | 2D-Streudiagramm mit wählbaren Komponenten für die X- und Y-Achse |
| **Residuen** | ✅ | ✅ | Rekonstruktionsfehler pro Spektrum zur Bewertung der Modellqualität |

### **3. Tipps** { #3-tips }

- Nutzen Sie den **Scree-Plot**, um die optimale Anzahl an Komponenten zu bestimmen (achten Sie auf den deutlichen Knick bzw. „Ellenbogen“ in der kumulativen Varianzkurve).
- Das Diagramm **Ladungen (Loadings)** zeigt, welche spektralen Merkmale am stärksten zur jeweiligen Komponente beitragen.
- Der Tab **Heatmap** visualisiert, wo sich der Score einer gewählten Komponente in den räumlichen `(x, y)`-Kartenkoordinaten konzentriert.
- Der Plot **Scores** macht Gruppierungen und Ausreißer in Ihrem Datensatz sichtbar.
- Der Tab **Residuen** hilft dabei, Spektren zu identifizieren, die vom Modell nur unzureichend rekonstruiert werden.
- Jeder Diagramm-Tab enthält eine **`matplotlib`-Symbolleiste** zum Zoomen, Verschieben und Speichern der Abbildungen.

-----

### **4. Zitation und Literaturhinweise** { #4-citations-and-references }

Wenn Sie diese MVA-Funktionen nutzen, um Ergebnisse für eine Publikation zu erzeugen, bitten wir Sie freundlich, die folgenden Fachartikel zu zitieren, welche die zugrunde liegenden Implementierungen und Kernalgorithmen beschreiben:

- **PCA-Algorithmus**: Wold, S., Esbensen, K., and Geladi, P., "Principal component analysis." *Chemometrics and intelligent laboratory systems* 2.1-3 (1987): 37-52.
- **NMF-Algorithmus**: Lee, Daniel D., and H. Sebastian Seung, "Learning the parts of objects by non-negative matrix factorization." *Nature* 401.6755 (1999): 788-791.
