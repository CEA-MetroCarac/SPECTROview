"""Comprehensive tests for SPECTROview internationalization (EN / FR)."""

from __future__ import annotations

import gc
import json

from PySide6.QtCore import QEvent, QLocale
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
)

import spectroview.main as main_mod
from spectroview.ai_agent.v_chat_panel import VChatPanel
from spectroview.api import settings as api_settings
from spectroview.i18n import (
    TranslationManager,
    bind_combo_items,
    bind_text,
    bind_tooltip,
    clear_dead_bindings,
    combo_data_or_text,
    get_language,
    normalize_language,
    set_combo_by_data_or_text,
    set_language,
    tr,
    tr_plural,
)
from spectroview.i18n.validator import (
    audit_untranslated_ui_literals,
    validate_catalog,
)
from spectroview.model.m_settings import MSettings
from spectroview.settings.ui.settings_dialog import SettingsDialog
from spectroview.view.components.common.v_notification import NotificationType


class TestTranslationCore:
    def test_normalize_language(self):
        assert normalize_language(None) == "en"
        assert normalize_language("") == "en"
        assert normalize_language("en") == "en"
        assert normalize_language("en_US") == "en"
        assert normalize_language("English") == "en"
        assert normalize_language("fr") == "fr"
        assert normalize_language("fr-FR") == "fr"
        assert normalize_language("Français") == "fr"
        assert normalize_language("de") == "en"

    def test_tr_and_formatting_in_en_and_fr(self):
        set_language("en", persist=False)
        assert tr("Open") == "Open"
        assert (
            tr("{peak} peak position (cm$^{{-1}}$)", peak="Si")
            == "Si peak position (cm$^{-1}$)"
        )

        set_language("fr", persist=False)
        assert tr("Open") == "Ouvrir"
        assert tr("Settings") == "Paramètres"
        assert (
            tr("{peak} peak position (cm$^{{-1}}$)", peak="Si")
            == "Position du pic Si (cm$^{-1}$)"
        )
        # Missing keys fall back gracefully to source text
        assert tr("__nonexistent_key_{x}__", x=42) == "__nonexistent_key_42__"
        # Missing format kwargs never raise
        assert tr("Graph {gid}", wrong_key=1) == "Graph {gid}"

    def test_tr_plural_english_vs_french_rules(self):
        # EN: only 1 is singular; 0 and >=2 are plural
        set_language("en", persist=False)
        assert tr_plural("{count} message", "{count} messages", 0) == "0 messages"
        assert tr_plural("{count} message", "{count} messages", 1) == "1 message"
        assert tr_plural("{count} message", "{count} messages", 2) == "2 messages"

        # FR: 0 and 1 are singular; >=2 is plural
        set_language("fr", persist=False)
        assert tr_plural("{count} message", "{count} messages", 0) == "0 message"
        assert tr_plural("{count} message", "{count} messages", 1) == "1 message"
        assert tr_plural("{count} message", "{count} messages", 2) == "2 messages"
        assert (
            tr_plural(
                "Created {count} wafer plot",
                "Created {count} wafer plots",
                1,
            )
            == "1 carte wafer créée"
        )
        assert (
            tr_plural(
                "Created {count} wafer plot",
                "Created {count} wafer plots",
                0,
            )
            == "0 carte wafer créée"
        )
        assert (
            tr_plural(
                "Created {count} wafer plot",
                "Created {count} wafer plots",
                3,
            )
            == "3 cartes wafer créées"
        )

    def test_context_lookup_and_fallback(self):
        mgr = TranslationManager.instance()
        set_language("fr", persist=False)
        mgr._catalog = {
            "strings": {"Open": "Ouvrir"},
            "contexts": {"ctx_test": {"Open": "Ouvrir le fichier"}},
            "plurals": {},
        }
        assert tr("Open", ctx="ctx_test") == "Ouvrir le fichier"
        assert tr("Open") == "Ouvrir"
        # Reset cached catalog cleanly
        mgr._catalog_cache.clear()
        set_language("en", persist=False)


class TestLanguagePersistence:
    def test_msettings_and_api_round_trip(self):
        settings = MSettings()
        assert settings.get_language() == "en"
        assert api_settings.get_language() == "en"

        api_settings.set_language("fr")
        assert MSettings().get_language() == "fr"
        assert api_settings.get_language() == "fr"

        api_settings.set_language("invalid_code")
        assert MSettings().get_language() == "en"

    def test_config_json_export_import_preserves_language(self, tmp_path):
        settings = MSettings()
        settings.set_language("fr")
        cfg_file = tmp_path / "config.json"
        exported = settings.export_config(str(cfg_file))
        assert exported == str(cfg_file)
        payload = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert payload["language"] == "fr"

        settings.set_language("en")
        assert settings.get_language() == "en"
        assert settings.import_config(str(cfg_file)) is True
        assert settings.get_language() == "fr"


class TestBindingsAndQtLocale:
    def test_weakref_bindings_and_gc_cleanup(self, qapp):
        lbl = QLabel()
        bind_text(lbl, "Open")
        bind_tooltip(lbl, "Settings")
        assert lbl.text() == "Open"
        assert lbl.toolTip() == "Settings"

        set_language("fr", persist=False)
        assert lbl.text() == "Ouvrir"
        assert lbl.toolTip() == "Paramètres"

        lbl.deleteLater()
        qapp.processEvents()
        del lbl
        gc.collect()

        # Switching language after widget destruction must never raise RuntimeError
        set_language("en", persist=False)
        clear_dead_bindings()

    def test_combo_canonical_item_data_preserved(self, qapp):
        combo = QComboBox()
        bind_combo_items(
            combo,
            [
                ("Light Mode", "Light Mode"),
                ("Dark Mode", "Dark Mode"),
            ],
        )
        assert combo.currentText() == "Light Mode"
        assert combo_data_or_text(combo) == "Light Mode"

        set_language("fr", persist=False)
        assert combo.itemText(0) == "Mode clair"
        assert combo.itemText(1) == "Mode sombre"
        assert combo_data_or_text(combo) == "Light Mode"

        assert set_combo_by_data_or_text(combo, "Dark Mode") is True
        assert combo.currentText() == "Mode sombre"
        assert combo_data_or_text(combo) == "Dark Mode"

        combo.close()
        combo.deleteLater()

    def test_qt_standard_buttons_and_dot_decimal_locale(self, qapp):
        set_language("fr", persist=False)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        spin = QDoubleSpinBox()
        # QLocale must stay dot-decimal so numeric inputs don't switch to comma decimals
        assert QLocale().decimalPoint() == "."
        assert spin.locale().decimalPoint() == "."

        mgr = TranslationManager.instance()
        # QTranslator loads qtbase_fr when shipped by PySide6
        if mgr._qt_translator is not None:
            cancel_btn = box.button(QDialogButtonBox.StandardButton.Cancel)
            assert cancel_btn.text().replace("&", "").lower() == "annuler"

        box.close()
        spin.close()
        box.deleteLater()
        spin.deleteLater()


class TestFrenchNotificationHeuristics:
    def test_infer_from_french_messages(self):
        assert (
            NotificationType.infer_from_message("Erreur lors du chargement.")
            == NotificationType.ERROR
        )
        assert (
            NotificationType.infer_from_message("Échec de l'exportation.")
            == NotificationType.ERROR
        )
        assert (
            NotificationType.infer_from_message("Aucun spectre sélectionné.")
            == NotificationType.WARNING
        )
        assert (
            NotificationType.infer_from_message("Veuillez patienter...")
            == NotificationType.WARNING
        )
        assert (
            NotificationType.infer_from_message("Modèle enregistré avec succès.")
            == NotificationType.SUCCESS
        )


class TestMainWindowLiveLanguageSwitching:
    def test_live_en_to_fr_to_en_manual_url_and_cancel_rollback(self, qapp, monkeypatch):
        opened_urls = []
        monkeypatch.setattr(
            QDesktopServices,
            "openUrl",
            lambda url: opened_urls.append(url.toString()) or True,
        )
        monkeypatch.setattr(main_mod.Main, "_prewarm_ai_chat", lambda self: None)

        win = main_mod.Main()
        try:
            initial_tab_count = win.tabWidget.count()
            initial_action_count = len(win.menu_bar.actions())

            assert get_language() == "en"
            assert win.tabWidget.tabText(0) == "Spectra"
            assert win.tabWidget.tabText(1) == "Maps"
            assert win.tabWidget.tabText(2) == "Graphs"
            assert win.menu_bar.actionOpen.text() == "Open"
            assert win.menu_bar.actionSettings.text() == "Settings"
            assert win.menu_bar.actionAIChat.text() == "SPECTROview AI Agent"
            assert win.v_spectra_workspace.btn_reinit.text() == "Reinit"
            assert win.v_maps_workspace.v_fit_model_builder.btn_fit.text() == "Fit"
            assert win.v_graphs_workspace.btn_add_plot.text() == "Add plot"

            win.manual()
            assert opened_urls[-1].endswith("/user_manual/")

            # Switch live to French
            win.set_language("fr")
            assert get_language() == "fr"
            assert (
                win.windowTitle()
                == "SPECTROview (Outil de traitement et de visualisation de données spectroscopiques)"
            )
            assert win.tabWidget.tabText(0) == "Spectres"
            assert win.tabWidget.tabText(1) == "Cartes"
            assert win.tabWidget.tabText(2) == "Graphiques"
            assert win.menu_bar.actionOpen.text() == "Ouvrir"
            assert win.menu_bar.actionSettings.text() == "Paramètres"
            assert win.menu_bar.actionAIChat.text() == "Agent IA SPECTROview"
            assert win.v_spectra_workspace.btn_reinit.text() == "Réinit."
            assert win.v_maps_workspace.v_fit_model_builder.btn_fit.text() == "Ajuster"
            assert win.v_graphs_workspace.btn_add_plot.text() == "Ajouter graphique"
            assert win.v_graphs_workspace.btn_apply_recipe.text() == "Recettes de tracé"

            win.manual()
            assert opened_urls[-1].endswith("/fr/user_manual/")

            # Switch back live to English — no duplicate tabs/actions, state intact
            win.set_language("en")
            assert get_language() == "en"
            assert win.tabWidget.count() == initial_tab_count
            assert len(win.menu_bar.actions()) == initial_action_count
            assert win.tabWidget.tabText(0) == "Spectra"
            assert win.menu_bar.actionOpen.text() == "Open"
            assert win.v_spectra_workspace.btn_reinit.text() == "Reinit"
            assert win.v_maps_workspace.v_fit_model_builder.btn_fit.text() == "Fit"
            assert win.v_graphs_workspace.btn_add_plot.text() == "Add plot"

            # Settings dialog live preview + Cancel rollback
            assert win.settings.get_language() == "en"
            preview_states = []

            def _fake_exec(dlg):
                idx = dlg.cbb_language.findData("fr")
                assert idx >= 0
                dlg.cbb_language.setCurrentIndex(idx)
                preview_states.append((get_language(), dlg.windowTitle()))
                return 0  # Cancel / Rejected

            monkeypatch.setattr(SettingsDialog, "exec", _fake_exec)
            win._open_settings()

            assert preview_states == [("fr", "Paramètres")]
            assert get_language() == "en"
            assert win.settings.get_language() == "en"
            assert win.tabWidget.tabText(0) == "Spectra"
            assert win.menu_bar.actionOpen.text() == "Open"
        finally:
            win.close()
            win.deleteLater()
            qapp.sendPostedEvents(None, QEvent.DeferredDelete)
            qapp.processEvents()
            gc.collect()

    def test_startup_restores_persisted_french_language(self, qapp, monkeypatch):
        monkeypatch.setattr(main_mod.Main, "_prewarm_ai_chat", lambda self: None)
        MSettings().set_language("fr")
        win = main_mod.Main()
        try:
            assert get_language() == "fr"
            assert win.tabWidget.tabText(0) == "Spectres"
            assert win.tabWidget.tabText(1) == "Cartes"
            assert win.tabWidget.tabText(2) == "Graphiques"
            assert win.menu_bar.actionOpen.text() == "Ouvrir"
            assert win.menu_bar.actionSettings.text() == "Paramètres"
        finally:
            win.close()
            win.deleteLater()
            qapp.sendPostedEvents(None, QEvent.DeferredDelete)
            qapp.processEvents()
            gc.collect()


class TestAIAgentLocalization:
    def test_chat_panel_live_switch_and_system_prompt_language_hint(self, qapp, monkeypatch):
        monkeypatch.setattr(VChatPanel, "_refresh_status", lambda self: None)
        panel = VChatPanel()
        try:
            idx_custom = panel.cbb_provider.findData("Custom")
            assert idx_custom >= 0
            panel.cbb_provider.setCurrentIndex(idx_custom)
            assert panel.btn_send.text() == "Send"
            assert panel.edit_title.placeholderText() == "New Conversation"

            en_prompt = panel.vm._build_system_prompt()
            assert "Respond to the user in French" not in en_prompt

            set_language("fr", persist=False)
            assert panel.btn_send.text() == "Envoyer"
            assert panel.edit_title.placeholderText() == "Nouvelle conversation"
            assert panel.cbb_provider.currentData() == "Custom"
            assert panel.cbb_provider.currentText() == "Personnalisé"

            fr_prompt = panel.vm._build_system_prompt()
            assert fr_prompt.startswith(en_prompt)
            assert "Respond to the user in French (français)" in fr_prompt

            set_language("en", persist=False)
            assert panel.btn_send.text() == "Send"
            assert panel.cbb_provider.currentData() == "Custom"
        finally:
            panel.close()
            panel.deleteLater()
            qapp.processEvents()


class TestCatalogAndSourceAudit:
    def test_fr_catalog_complete_and_valid(self):
        report = validate_catalog("fr")
        assert report.ok, (
            f"Catalog validation failed: missing_strings={report.missing_strings}, "
            f"empty_strings={report.empty_strings}, "
            f"missing_plurals={report.missing_plurals}, "
            f"empty_plurals={report.empty_plurals}, "
            f"missing_contexts={report.missing_contexts}, "
            f"placeholder_mismatches={report.placeholder_mismatches}"
        )

    def test_no_untranslated_ui_literals(self):
        findings = audit_untranslated_ui_literals()
        assert findings == [], "Untranslated UI string literals detected:\n" + "\n".join(
            findings
        )
