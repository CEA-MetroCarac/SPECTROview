## **Arbeitsbereich speichern & laden** { #save-load-workspace }

**SPECTROview** verwendet eigene native Dateiformate, damit Sie Ihre Datenanalysesitzungen nahtlos unterbrechen, speichern und fortsetzen können, ohne dass Konfigurationen oder Fit-Ergebnisse verloren gehen.

### **1. Ihre Arbeit speichern** { #1-saving-your-work }

Um Ihren aktuellen Arbeitsstand zu speichern, klicken Sie in der Hauptsymbolleiste der Anwendung auf die Schaltfläche **Speichern** (Save):

<div align="center">
  <img src="../user_manual_images/Save_Load/save_load_clear.png" alt="Schaltflächen Speichern, Laden und Leeren" width="120"><br>
</div>

Die Anwendung erzeugt je nach aktuell aktivem Arbeitsbereich eine spezifische Dateiendung:

| Aktiver Arbeitsbereich | Native Dateiendung | Gespeicherte Inhalte |
|---|---|---|
| **Spektren** (Spectra) | `.spectra` | Alle geladenen 1D-Spektren, benutzerdefinierten `baseline`-Konfigurationen, mathematischen Fit-Modelle und zusammengefassten Fit-Ergebnisse. |
| **Karten** (Maps) | `.maps` | Die Rohdaten der 2D-/Wafer-Karte, Gerätemetadaten, alle zugehörigen Spektren und alle punktweisen Fit-Ergebnisse. |
| **Diagramme** (Graphs) | `.graphs` | Alle geladenen mathematischen Datensätze, individuell angepassten Plot-Konfigurationen und gezeichneten Anmerkungen. |

### **2. Ihre Arbeit laden** { #2-loading-your-work }

Um eine frühere Sitzung fortzusetzen, klicken Sie einfach in der Hauptsymbolleiste auf die Schaltfläche **Öffnen** (Open) und wählen Sie eine zuvor gespeicherte `.spectra`-, `.maps`- oder `.graphs`-Datei aus. **SPECTROview** erkennt das Dateiformat automatisch, stellt Ihren Datensatz exakt so wieder her, wie Sie ihn verlassen haben, und öffnet sofort den Tab des entsprechenden Arbeitsbereichs.
