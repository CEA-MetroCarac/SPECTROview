## **Einstellungen & Präferenzen** { #settings-preferences }

Öffnen Sie das Menü **Einstellungen** (Settings) über das Zahnradsymbol in der Hauptsymbolleiste (oder drücken Sie `Ctrl + Shift + S`). Sie können auch in der oberen Menüleiste auf **Über SPECTROview** (About SPECTROview) klicken, um direkt zum Abschnitt **Über** (About) zu springen.

<div align="center">
  <img src="../user_manual_images/Settings/settings.png" alt="Einstellungsfenster" width="820"><br>
</div>

<br>

Das `Settings Panel` bietet eine moderne zweigeteilte Benutzeroberfläche mit Echtzeit-Suchfilterung:

- **Linke Seitenleiste**:
  - **Suchleiste**: Geben Sie ein beliebiges Stichwort ein (z. B. `working folder`, `theme`, `language`, `tolerance`, `loss`, `mcp`, `python`, `citation`), um die Navigationsliste sofort zu filtern und passende Optionen direkt auf der Seite mit automatischem Scrollen hervorzuheben.
  - **Navigations-Tabs**:
    1. **Allgemein** (General) — SPECTROview-Arbeitsordner, Oberflächen-Design, Oberflächensprache, portables Konfigurations-Backup bzw. Wiederherstellung sowie Update-Einstellungen.
    2. **Fit-Einstellungen** (Fit Settings) — Konvergenzparameter des Optimierers, robuste Verlustfunktion, Peak-Polarität und globale Peak-Grenzen (mit eigenem Symbol für Kurvenanpassung).
    3. **KI-Einstellungen** (AI Settings) — Zugangsdaten für LLM-Anbieter, Verbindungstest und lokaler MCP-Server.
    4. **Über** — Anwendungsversion, Build-Nummer, Python-Umgebung, Versionen der Laufzeitabhängigkeiten, Dokumentation und Zitation.
- **Rechter Inhaltsbereich**: Zeigt alle Optionen des ausgewählten Tabs an. Sämtliche Hinweistexte, Umgebungsdetails und Zitationen lassen sich zum einfachen Kopieren mit der Maus markieren.
- **Fußzeile**: Zeigt die Anwendungsversion zusammen mit den Standardschaltflächen **Speichern** (Save) und **Abbrechen** (Cancel) an.

> **Design-Synchronisierung**: Das **Oberflächen-Design** (Interface Theme) kann entweder im Tab **Allgemein** (`Modern Dark`, `Modern Gray`, `Modern Light`, `Classic Dark` oder `Classic Light`) oder direkt über die Schaltfläche **Design** (Theme) in der oberen Menüleiste ausgewählt werden. Beide Auswahlfelder werden in Echtzeit vollständig synchronisiert.

---

### **1. SPECTROview-Arbeitsordner & Präferenzen** *(Tab Allgemein)* { #1-spectroview-working-folder-preferences-general-tab }

Legen Sie einen einzigen universellen Stammordner für sämtliche Vorlagen, Peak-Profile, Rezepte, Stile, KI-Chat-Verläufe und Konfigurationssicherungen von SPECTROview fest. Sobald Sie ihn angeben, werden automatisch sechs Unterordner und eine portable Konfigurationsdatei darin erstellt:

| Unterordner / Datei | Inhalt |
|---------------------|--------|
| `fit_model/` | Benutzerdefinierte Fit-Modelle (JSON). Der **Fit-Modell-Editor** (`Fit Model Builder`) durchsucht diesen Ordner und listet jedes Modell zur Wiederverwendung per Klick auf. |
| `peak_profile/` | Benutzerdefinierte globale Peak-Grenzprofile (JSON) mit gespeicherten `maxshift`-, `minfwhm`- und `maxfwhm`-Vorgaben je Messtechnik. |
| `plot_recipe/` | **Plot-Rezepte** (Plot Recipes) — gespeicherte Sätze vollständiger Plot-Konfigurationen (Datenbindungen + Stil) für den Arbeitsbereich **Diagramme** (`Graphs`). |
| `plot_style/` | **Stilvorlagen** (Style Templates) — gespeicherte reine Erscheinungsbild-Stile, die auf jedes vorhandene Diagramm angewendet werden können. |
| `ai_chat_history/` | Protokolle der KI-Chat-Unterhaltungen (JSON). Gespräche werden hier automatisch gespeichert. |
| `template/` | Allgemeine Plot- und Arbeitsbereichsvorlagen. |
| `config.json` | Portables Konfigurations-Backup. Nicht-geheime Anwendungseinstellungen werden in diese JSON-Datei gespiegelt. |

- Klicken Sie auf **Durchsuchen** (Browse), um einen Ordner auszuwählen, oder geben Sie den Pfad direkt ein.
- Klicken Sie auf **Öffnen** (Open), um den konfigurierten Ordner mit einem Klick direkt im **Windows-Explorer** oder **macOS Finder** zu öffnen.

> **Sicherheit der Zugangsdaten**: In den Einstellungen eingegebene API-Schlüssel für KI-Anbieter werden unter Windows, macOS und Linux im benutzerspezifischen privaten Anwendungsspeicher Ihres Betriebssystems abgelegt (`QSettings` unter einem internen obfuskierten Namensraum) und strikt aus `config.json`-Sicherungen ausgeschlossen. Dadurch werden störende Abfragen von OS Keychain / KWallet vermieden und zugleich verhindert, dass sensible Zugangsdaten offengelegt werden, wenn Sie Ihren Arbeitsordner teilen oder versionieren.
>
> **Datensicherung & PC-Migration**: Wenn Sie auf einen neuen PC wechseln oder eine Sicherung erstellen möchten, kopieren Sie einfach diesen einen Arbeitsordner auf den neuen Rechner. Wählen Sie dort im Einstellungsfenster von SPECTROview denselben Ordner als **SPECTROview-Arbeitsordner** (SPECTROview Working Folder) aus: Alle Fit-Modelle, Plot-Rezepte, Stilvorlagen, KI-Chat-Verläufe sowie die Konfiguration (`config.json`) werden automatisch wiederhergestellt!
>
> Darüber hinaus können Sie jederzeit über **Konfig. exportieren...** (Export Config...) und **Konfig. wiederherstellen...** (Restore Config...) Einstellungen in eine eigenständige JSON-Datei exportieren oder daraus wiederherstellen.

#### **Oberflächen-Design** { #interface-theme }

- **Design**: Wählen Sie zwischen **Modern Dunkel** (Modern Dark), **Modern Grau** (Modern Gray, Standard), **Modern Hell** (Modern Light), **Klassisch Dunkel** (Classic Dark) oder **Klassisch Hell** (Classic Light). Die Auswahl aktualisiert das aktive Design sofort und hält die Design-Schaltfläche in der oberen Menüleiste vollständig synchron. Das Design der Plot-Zeichenfläche ist vom Oberflächen-Design unabhängig und verwendet standardmäßig die Option **Heller Modus** (Light Mode); Sie können es jederzeit über das Auswahlfeld **Plot-Theme** (Plot theme) in der Werkzeugleiste `View Options` unter dem Spektrum-Canvas auf **Dunkler Modus** (Dark Mode) umstellen (Ihre Auswahl bleibt sitzungsübergreifend erhalten).

#### **Oberflächensprache** { #interface-language }

- **Sprache** (Language): Wählen Sie zwischen **English** (Standard), **Français**, **简体中文** (vereinfachtes Chinesisch), **Deutsch** und **Русский** (Russisch). Änderungen werden **sofort in der gesamten Anwendung** wirksam — in Menüs, Arbeitsbereichen, Viewern, Dialogen, wissenschaftlichen Plot-Beschriftungen und im KI-Chat-Bereich — ohne dass ein Neustart erforderlich ist. Ein Klick auf **Abbrechen** setzt eine im Live-Modus getestete Sprachänderung auf die zuvor gespeicherte Einstellung zurück. Die gewählte Sprache bleibt sitzungsübergreifend über `QSettings` und `config.json` gespeichert, und ein Klick auf **Benutzerhandbuch öffnen** (Open User Manual) öffnet die entsprechende Sprachversion der Dokumentation (`/user_manual/` für Englisch, `/fr/user_manual/` für Französisch, `/de/user_manual/` für Deutsch, `/ru/user_manual/` für Russisch und `/zh/user_manual/` für vereinfachtes Chinesisch).

#### **Anwendungs-Updates** { #application-updates }

- **Beim Programmstart automatisch nach Updates suchen** (Check for updates automatically on application startup): Wenn diese Option aktiviert ist (Standard), prüft SPECTROview beim Start regelmäßig im Hintergrund auf GitHub Releases (mit HTTP-`ETag`-Caching) und berücksichtigt dabei Ihr Installationsformat (`pip`, Windows-Installer, portables ZIP-Archiv, macOS-DMG, Linux-AppImage oder `.deb`).
- **Vorabversionen (Beta / Release Candidate) einbeziehen** (Include pre-release / release candidate builds): Wenn diese Option aktiviert ist, berücksichtigen Update-Prüfungen auch Vorabversions-Tags (wie `-rc.1` oder `-beta`). Standardmäßig deaktiviert, sodass die meisten Nutzer nur stabile Versionen erhalten.

---

### **2. Fit-Konvergenzparameter** *(Tab Fit-Einstellungen)* { #2-fit-convergence-parameters-fit-settings-tab }

Diese Parameter steuern das Verhalten der **Vektorisierten Batch-Fit-Engine (`VBF`)** — des mathematischen Optimierers, der die Kurvenanpassung Ihrer Spektren ausführt. Eine Anpassung dieser Werte wirkt sich direkt auf Berechnungsgeschwindigkeit, Genauigkeit und Konvergenz aus.

- **Auf Standard zurücksetzen** (Reset to Default): Klicken Sie oben im Tab auf **Alle Fit-Einstellungen zurücksetzen** (Reset all fit settings to default), um sämtliche Konvergenzparameter und globalen Peak-Grenzen sofort auf die Werkseinstellungen zurückzusetzen.

| Parameter | Einstellung | Standard | Bereich | Beschreibung & Verhalten |
|-----------|-------------|----------|---------|--------------------------|
| **Maximale Iterationen** | `max_ite` | 200 | 10 – 10 000 | Maximale Anzahl an Levenberg-Marquardt-Iterationen pro Spektrum. <br>• **Niedriger (50–100)**: Schnelleres Fitting für Vorschauen.<br>• **Höher (200+)**: Bessere Konvergenz bei komplexen Spektren. |
| **x-Toleranz** | `xtol` | 1×10⁻⁴ | 1×10⁻¹⁰ – 1.0 | Relative Toleranz für die Schrittweite der Parameter. Konvergenz erfordert: <br> mean(|&delta;<i>p</i>| / |<i>p</i>|) &lt; `xtol` |
| **f-Toleranz** | `ftol` | 1×10⁻⁴ | 1×10⁻¹⁰ – 1.0 | Relative Toleranz für die Kostenfunktion. Konvergenz erfordert: <br> |cost<sub>n</sub> - cost<sub>n-1</sub>| / |cost<sub>n</sub>| &lt; `ftol` |
| **Rauschschwellen-Koeffizient** | `coef_noise` | 0.0 | 0.0 – 10.0 | Multiplikator für das automatisch geschätzte Rauschniveau (`coef_noise` × geschätzte Rauschamplitude). Maskiert verrauschte Daten und unterdrückt Geister-Peaks.<br>• **0.0**: Deaktiviert (Standard)<br>• **0.5–1.0**: Konservativ<br>• **1.0–2.0**: Moderat<br>• **3.0+**: Aggressiv<br><br>*Hinweis: Sie können das in der Symbolleiste angezeigte Rauschniveau ablesen oder das Kontrollkästchen **Rauschpegel anzeigen** (Show noise level) in den `View Options` des `SpectraViewer` aktivieren, um den Rauschpegel direkt im Spektren-Plot anzuzeigen.* |
| **Verlustfunktion** | `loss` | `linear` | `linear`, `soft_l1`, `huber` | Robuste Verlustfunktion für Ausreißerresistenz (z. B. Spikes durch kosmische Strahlung oder scharfe Artefakte). <br>• **linear**: Standardmäßige Methode der kleinsten Quadrate (Standard).<br>• **soft_l1**: Glatte L1-Approximation $\rho(z) = 2(\sqrt{1+z}-1)$, robust gegenüber starken Spikes ohne manuelle Maskierung.<br>• **huber**: Huber-Verlustfunktion, quadratisch für gültige Punkte und linear für Residuen oberhalb von `f_scale`. |
| **Verlustskala** | `f_scale` | 1.0 | 0.01 – 100.0 | Grenzwertskala zwischen regulären Punkten und Ausreißern bei robusten Verlustfunktionen. Residuen werden vor der Auswertung von $\rho'(z)$ als $(r / f\_scale)$ skaliert. |
| **Negative Werte fitten** | `fit_negative` | Deaktiviert | - | Wenn aktiviert, werden negative Intensitätswerte (Absorptions- oder invertierte Peaks) in den Fit einbezogen. Wenn deaktiviert, erhalten sie das Gewicht null. |

> **Hinweis**: Ein Spektrum gilt erst dann als konvergiert, wenn **beide** Kriterien, `xtol` und `ftol`, gleichzeitig erfüllt sind.
>
> **Reihenfolge der Gewichtungsberechnung**: Harte Ausschlüsse (`fit_negative` und die Rauschuntergrenze `coef_noise`) werden zuerst ausgewertet, um ausgeschlossenen Punkten das Gewicht null zuzuweisen. Für aktive Punkte wird anschließend die robuste Verlustgewichtung (`soft_l1` oder `huber`) angewendet, wodurch Ausreißer-Residuen über *Iteratively Reweighted Least Squares* (IRLS) sanft herabgewichtet werden.

---

### **3. Globale Peak-Grenzen** *(Tab Fit-Einstellungen)* { #3-global-peak-limits-fit-settings-tab }

Diese Parameter legen **absolute Randbedingungen** fest, die anwendungsweit für alle Peaks gelten. Wenn Sie einem Fit-Modell einen neuen Peak hinzufügen, werden diese Grenzwerte automatisch zur Initialisierung der Parameterbereiche verwendet. Über das Zahnrad-Symbol in der Gruppe `Peaks` von `Fit Model Builder` können Sie direkt in diesen Bereich springen.

- **Peak-Profil** (Peak profile:): Wechseln Sie zwischen integrierten Technik-Voreinstellungen (**Raman (cm⁻¹)**, **PL (eV)**, **PL (nm)**, **XRD (°)**) oder Ihren eigenen gespeicherten Profilen.
- **Profil speichern / löschen**: Klicken Sie auf das Speichern-Symbol neben **Peak-Profil**, um die aktuellen Werte für `Max peak shift`, `Min peak FWHM`, `Max peak FWHM` und `Default peak FWHM` unter einem eigenen Namen in `<Arbeitsordner>/peak_profile/` zu speichern. Wählen Sie ein benutzerdefiniertes Profil aus und klicken Sie auf das Papierkorb-Symbol, um es zu löschen. Eine manuelle Änderung der Grenzwerte schaltet die Auswahl auf **Benutzerdefiniert** (Custom).

> **Hinweis**: Sie können das Kontrollkästchen **Grenzen** (Limits) unterhalb von `PeakTable` aktivieren, um die tatsächlichen Grenzen jedes Peaks anzuzeigen.

| Grenzwert | Standard | Bereich | Beschreibung & Einschränkung |
|-----------|----------|---------|------------------------------|
| **Max. Peak-Verschiebung** | 20 | 0 – 10 000 | Maximal zulässige Verschiebung der Peak-Mitte (`x0`) gegenüber ihrem Startwert. <br><i>x</i><sub>0</sub> - maxshift &le; <i>x</i><sub>0</sub><sup>fitted</sup> &le; <i>x</i><sub>0</sub> + maxshift |
| **Min. Peak-FWHM** | 0.1 | 0.0001 – 10 000 | Minimal zulässige Halbwertsbreite (`FWHM`). Verhindert, dass Peaks zu Delta-Funktions-Artefakten kollabieren. |
| **Max. Peak-FWHM** | 200 | 0.001 – 10 000 | Maximal zulässige `FWHM`. Verhindert, dass sich Peaks zu flachen Basislinien verbreitern. |
| **Standard-Peak-FWHM** (Default peak FWHM:) | 5.0 | 0.0001 – 10 000 | Anfängliche `FWHM` beim Hinzufügen eines neuen Peaks per Klick im `SpectraViewer` (z. B. `5.0` für Raman, `0.05` für PL (eV)). |

---

### **4. KI-Einstellungen** *(Tab KI-Einstellungen)* { #4-ai-settings-ai-settings-tab }

Der Tab **KI-Einstellungen** konfiguriert den [KI-Chat-Agenten](ai_agent.md). Für die Nutzung der übrigen Funktionen von SPECTROview ist hier keine Eingabe erforderlich.

#### **API-Schlüssel** { #api-keys }

Standardmäßig werden die Felder für einen Endpunkt vom Typ **Benutzerdefiniert** (Custom; OpenAI-kompatibel) angezeigt:

| Feld | Beschreibung |
|------|--------------|
| **Benutzerdefinierter API-Schlüssel** (Custom API Key) | Der API-Schlüssel für Ihren benutzerdefinierten oder internen Endpunkt. |
| **Basis-URL** (Base URL) | Die OpenAI-kompatible Stamm-URL des Endpunkts (z. B. `https://host/v1`). |
| **Modellname(n)** (Model Name) | Eine kommagetrennte Liste von Modellnamen (z. B. `model-a, model-b`). Diese füllen das Modell-Dropdown im Chat-Panel — praktisch für Endpunkte, die keine API zum Auflisten verfügbarer Modelle bereitstellen. |
| **Anzeigename** (Display Label) | Beschriftung, die im KI-Chat-Eingabebereich für diesen benutzerdefinierten Endpunkt angezeigt wird. |

Die Schlüssel für integrierte Cloud-Anbieter befinden sich im einklappbaren Abschnitt **▸ Weitere Cloud-Anbieter** (Other cloud providers) (zum Aufklappen klicken): **OpenAI**, **Anthropic**, **Gemini**, **DeepSeek** und **Mistral**. Alle Schlüssel werden maskiert gespeichert und sitzungsübergreifend gemerkt.

- **Verbindung testen** (Test Connection): Wählen Sie einen beliebigen Anbieter aus dem Dropdown-Menü aus und klicken Sie auf **Verbindung testen**, um Verfügbarkeit des API-Endpunkts, Zugangsdaten und Netzwerkverbindung ohne Speichern zu überprüfen.

> **Speicherort Ihrer Schlüssel**: API-Schlüssel werden im benutzerspezifischen
> Einstellungsspeicher Ihres Betriebssystems abgelegt (obfuskiert unter `_credentials/`) — in der Registrierung
> (`HKEY_CURRENT_USER\Software\CEA-Leti\SPECTROview`) unter Windows,
> unter `~/Library/Preferences` auf macOS bzw. in `~/.config/CEA-Leti/SPECTROview.conf` unter Linux.
> Sie werden **niemals** in eine Datei innerhalb des SPECTROview-Projektordners geschrieben,
> sind weder in exportierten Arbeitsbereichen noch in Konfigurationssicherungen enthalten und können nicht
> versehentlich in ein Git-Repository committet werden.

#### **Lokaler MCP-Server** { #local-mcp-server }

Aktivieren Sie diese Option nur, wenn ein vertrauenswürdiger MCP-Client auf demselben Computer Zugriff auf die aktuell laufende SPECTROview-Sitzung benötigt. Der Standard-Endpunkt lautet `http://127.0.0.1:8765/mcp` und startet, sobald Sie **Speichern** wählen.

| Feld | Standard | Beschreibung |
|---|---|---|
| **Lokalen Model-Context-Protocol-(MCP)-Endpunkt aktivieren** (Enable local MCP endpoint) | Aus | Startet oder stoppt den integrierten MCP-Server. |
| **Port** | 8765 | Lokaler Streamable-HTTP-Port (1024–65535). Wählen Sie einen anderen Port, falls dieser belegt ist. |

Die Bind-Adresse ist fest auf Localhost (`127.0.0.1`) eingestellt; der Server wird weder im lokalen Netzwerk noch im Internet freigegeben. Lassen Sie SPECTROview geöffnet, solange der Client verbunden ist.

---

### **5. Über & Umgebung** *(Tab Über)* { #5-about-environment-about-tab }

Der Tab **Über** bietet umfassende Laufzeit- und Referenzinformationen:

- **Anwendungsdetails**: Offizielles SPECTROview-Logo, Titel, Versionsnummer (`Version {VERSION}`) und Build-Nummer (`Build: {BUILD_NUMBER}`).
- **Umgebung & Abhängigkeiten** (Environment & Dependencies): Erkennt automatisch den Ausführungs- und Paketierungsmodus (`Python Package (.whl / local)`, `Windows Installer (Velopack)`, `Windows Portable`, `macOS Application (.app)`, `Linux AppImage` oder `Linux Debian Package (.deb)`) und zeigt die Versionen der Abhängigkeiten an:

  - **Eigenständige Builds (Standalone)**: Zeigen den bei der Paketierung eingefrorenen Abhängigkeits-Snapshot sowie Details zum integrierten Python und zum Host-/Build-Betriebssystem an.
  - **Installiertes Paket (`.whl` / `pip` / Dev)**: Ermittelt dynamisch die aktiven Paketversionen in der lokalen Python-Umgebung.
  - Listet `PySide6` (mit Qt-Laufzeitversion), TLS- und Sicherheits-Stack (`certifi`, `truststore`), wissenschaftlichen Stack (`numpy`, `scipy`, `matplotlib`, `pandas`, `pybaselines`), Export-Engines (`openpyxl`, `pyarrow`), UI-Erweiterungen (`superqt`) sowie KI-Chat-SDKs auf.
- **Nach Updates suchen** (Check for Updates): Direkte Aktion zur Prüfung auf neue Versionen über GitHub Releases.
- **Artikelzitation**: Die kanonische begutachtete *SoftwareX*-Zitation mit DOI sowie eine Schaltfläche **Kopieren** (Copy), um die Zitation schnell in die Zwischenablage zu kopieren.
