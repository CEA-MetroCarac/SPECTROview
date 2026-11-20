"""Comprehensive tests for SPECTROview multilingual internationalization (EN / FR / ZH / DE / RU)."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest
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
    SUPPORTED_LANGUAGES,
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
    def test_supported_languages_native_names(self):
        assert SUPPORTED_LANGUAGES == (
            ("en", "English"),
            ("fr", "Français"),
            ("zh", "简体中文"),
            ("de", "Deutsch"),
            ("ru", "Русский"),
        )

    def test_normalize_language(self):
        assert normalize_language(None) == "en"
        assert normalize_language("") == "en"
        assert normalize_language("en") == "en"
        assert normalize_language("en_US") == "en"
        assert normalize_language("English") == "en"
        assert normalize_language("fr") == "fr"
        assert normalize_language("fr-FR") == "fr"
        assert normalize_language("Français") == "fr"
        assert normalize_language("zh") == "zh"
        assert normalize_language("zh_CN") == "zh"
        assert normalize_language("zh-CN") == "zh"
        assert normalize_language("zh-Hans") == "zh"
        assert normalize_language("简体中文") == "zh"
        assert normalize_language("de") == "de"
        assert normalize_language("de_DE") == "de"
        assert normalize_language("de-DE") == "de"
        assert normalize_language("Deutsch") == "de"
        assert normalize_language("ru") == "ru"
        assert normalize_language("ru_RU") == "ru"
        assert normalize_language("ru-RU") == "ru"
        assert normalize_language("Русский") == "ru"
        assert normalize_language("xx_UNKNOWN") == "en"

    @pytest.mark.parametrize(
        "lang, expected_open, expected_settings, expected_peak",
        [
            ("en", "Open", "Settings", "Si peak position (cm$^{-1}$)"),
            ("fr", "Ouvrir", "Paramètres", "Position du pic Si (cm$^{-1}$)"),
            ("zh", "打开", "设置", "Si 峰位 (cm⁻¹)"),
            ("de", "Öffnen", "Einstellungen", "Si-Peakposition (cm$^{-1}$)"),
            ("ru", "Открыть", "Настройки", "Положение пика Si (см$^{-1}$)"),
        ],
    )
    def test_tr_and_formatting_in_all_languages(
        self, lang, expected_open, expected_settings, expected_peak
    ):
        set_language(lang, persist=False)
        assert tr("Open") == expected_open
        assert tr("Settings") == expected_settings
        assert tr("{peak} peak position (cm$^{{-1}}$)", peak="Si") == expected_peak
        # Missing keys fall back gracefully to source text
        assert tr("__nonexistent_key_{x}__", x=42) == "__nonexistent_key_42__"
        # Missing format kwargs never raise
        assert tr("Graph {gid}", wrong_key=1) == "Graph {gid}"

    def test_tr_plural_rules_across_languages(self):
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
            tr_plural("Created {count} wafer plot", "Created {count} wafer plots", 1)
            == "1 carte wafer créée"
        )
        assert (
            tr_plural("Created {count} wafer plot", "Created {count} wafer plots", 0)
            == "0 carte wafer créée"
        )
        assert (
            tr_plural("Created {count} wafer plot", "Created {count} wafer plots", 3)
            == "3 cartes wafer créées"
        )

        # ZH: invariant single plural form
        set_language("zh", persist=False)
        assert tr_plural("{count} spectrum", "{count} spectra", 0) == "0 条光谱"
        assert tr_plural("{count} spectrum", "{count} spectra", 1) == "1 条光谱"
        assert tr_plural("{count} spectrum", "{count} spectra", 5) == "5 条光谱"

        # DE: 1 singular; 0 and >=2 plural
        set_language("de", persist=False)
        assert tr_plural("{count} spectrum", "{count} spectra", 0) == "0 Spektren"
        assert tr_plural("{count} spectrum", "{count} spectra", 1) == "1 Spektrum"
        assert tr_plural("{count} spectrum", "{count} spectra", 5) == "5 Spektren"

        # RU: 3 Slavic forms (1/21 -> 0; 2-4/22-24 -> 1; 0/5-20/25 -> 2)
        set_language("ru", persist=False)
        assert tr_plural("{count} spectrum", "{count} spectra", 1) == "1 спектр"
        assert tr_plural("{count} spectrum", "{count} spectra", 21) == "21 спектр"
        assert tr_plural("{count} spectrum", "{count} spectra", 2) == "2 спектра"
        assert tr_plural("{count} spectrum", "{count} spectra", 24) == "24 спектра"
        assert tr_plural("{count} spectrum", "{count} spectra", 0) == "0 спектров"
        assert tr_plural("{count} spectrum", "{count} spectra", 5) == "5 спектров"
        assert tr_plural("{count} spectrum", "{count} spectra", 11) == "11 спектров"
        assert tr_plural("{count} spectrum", "{count} spectra", 14) == "14 спектров"
        assert tr_plural("{count} spectrum", "{count} spectra", 25) == "25 спектров"

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
    @pytest.mark.parametrize("lang", ["fr", "zh", "de", "ru"])
    def test_msettings_and_api_round_trip(self, lang):
        settings = MSettings()
        assert settings.get_language() == "en"
        assert api_settings.get_language() == "en"

        api_settings.set_language(lang)
        assert MSettings().get_language() == lang
        assert api_settings.get_language() == lang

        api_settings.set_language("invalid_code")
        assert MSettings().get_language() == "en"

    @pytest.mark.parametrize("lang", ["fr", "zh", "de", "ru"])
    def test_config_json_export_import_preserves_language(self, tmp_path, lang):
        settings = MSettings()
        settings.set_language(lang)
        cfg_file = tmp_path / "config.json"
        exported = settings.export_config(str(cfg_file))
        assert exported == str(cfg_file)
        payload = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert payload["language"] == lang

        settings.set_language("en")
        assert settings.get_language() == "en"
        assert settings.import_config(str(cfg_file)) is True
        assert settings.get_language() == lang


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

    @pytest.mark.parametrize(
        "lang, expected_cancel",
        [
            ("fr", "annuler"),
            ("zh", "取消"),
            ("de", "abbrechen"),
            ("ru", "отмена"),
        ],
    )
    def test_qt_standard_buttons_and_dot_decimal_locale(self, qapp, lang, expected_cancel):
        set_language(lang, persist=False)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        spin = QDoubleSpinBox()
        # QLocale must stay dot-decimal so numeric inputs don't switch to comma decimals
        assert QLocale().decimalPoint() == "."
        assert spin.locale().decimalPoint() == "."

        mgr = TranslationManager.instance()
        # QTranslator loads qtbase_<lang> when shipped by PySide6
        if mgr._qt_translator is not None:
            cancel_btn = box.button(QDialogButtonBox.StandardButton.Cancel)
            btn_text = cancel_btn.text().replace("&", "").lower()
            assert expected_cancel in btn_text or btn_text == "cancel"

        box.close()
        spin.close()
        box.deleteLater()
        spin.deleteLater()


class TestNotificationHeuristics:
    @pytest.mark.parametrize(
        "msg, expected",
        [
            # French
            ("Erreur lors du chargement.", NotificationType.ERROR),
            ("Échec de l'exportation.", NotificationType.ERROR),
            ("Aucun spectre sélectionné.", NotificationType.WARNING),
            ("Veuillez patienter...", NotificationType.WARNING),
            ("Modèle enregistré avec succès.", NotificationType.SUCCESS),
            # Chinese
            ("加载文件时出错。", NotificationType.ERROR),
            ("未选择光谱。", NotificationType.WARNING),
            ("拟合模型保存成功。", NotificationType.SUCCESS),
            # German
            ("Fehler beim Laden.", NotificationType.ERROR),
            ("Kein Spektrum ausgewählt.", NotificationType.WARNING),
            ("Modell erfolgreich gespeichert.", NotificationType.SUCCESS),
            # Russian
            ("Ошибка загрузки файла.", NotificationType.ERROR),
            ("Нет выбранного спектра.", NotificationType.WARNING),
            ("Модель успешно сохранена.", NotificationType.SUCCESS),
        ],
    )
    def test_infer_from_multilingual_messages(self, msg, expected):
        assert NotificationType.infer_from_message(msg) == expected


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

            # Switch live through French, Chinese, German, Russian, and back to English
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

            # Chinese
            win.set_language("zh")
            assert get_language() == "zh"
            assert win.tabWidget.tabText(0) == "光谱"
            assert win.tabWidget.tabText(1) == "二维图谱"
            assert win.tabWidget.tabText(2) == "图表"
            assert win.menu_bar.actionOpen.text() == "打开"
            assert win.menu_bar.actionSettings.text() == "设置"
            assert win.v_spectra_workspace.btn_reinit.text() == "重置"
            assert win.v_maps_workspace.v_fit_model_builder.btn_fit.text() == "拟合"
            win.manual()
            assert opened_urls[-1].endswith("/user_manual/") and "/fr/" not in opened_urls[-1]

            # German
            win.set_language("de")
            assert get_language() == "de"
            assert win.tabWidget.tabText(0) == "Spektren"
            assert win.tabWidget.tabText(1) == "Karten"
            assert win.tabWidget.tabText(2) == "Diagramme"
            assert win.menu_bar.actionOpen.text() == "Öffnen"
            assert win.menu_bar.actionSettings.text() == "Einstellungen"
            assert win.v_spectra_workspace.btn_reinit.text() == "Zurücksetzen"
            assert win.v_maps_workspace.v_fit_model_builder.btn_fit.text() == "Anpassen"
            win.manual()
            assert opened_urls[-1].endswith("/user_manual/") and "/fr/" not in opened_urls[-1]

            # Russian
            win.set_language("ru")
            assert get_language() == "ru"
            assert win.tabWidget.tabText(0) == "Спектры"
            assert win.tabWidget.tabText(1) == "Карты"
            assert win.tabWidget.tabText(2) == "Графики"
            assert win.menu_bar.actionOpen.text() == "Открыть"
            assert win.menu_bar.actionSettings.text() == "Настройки"
            assert win.v_spectra_workspace.btn_reinit.text() == "Сброс"
            assert win.v_maps_workspace.v_fit_model_builder.btn_fit.text() == "Подгонка"
            win.manual()
            assert opened_urls[-1].endswith("/user_manual/") and "/fr/" not in opened_urls[-1]

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

            # Settings dialog exposes all 5 native language names + Cancel rollback
            assert win.settings.get_language() == "en"
            preview_states = []

            def _fake_exec(dlg):
                combo_labels = [dlg.cbb_language.itemText(i) for i in range(dlg.cbb_language.count())]
                assert combo_labels == ["English", "Français", "简体中文", "Deutsch", "Русский"]
                idx = dlg.cbb_language.findData("zh")
                assert idx >= 0
                dlg.cbb_language.setCurrentIndex(idx)
                preview_states.append((get_language(), dlg.windowTitle()))
                return 0  # Cancel / Rejected

            monkeypatch.setattr(SettingsDialog, "exec", _fake_exec)
            win._open_settings()

            assert preview_states == [("zh", "设置")]
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

    @pytest.mark.parametrize(
        "lang, tab0, action_open, action_settings",
        [
            ("fr", "Spectres", "Ouvrir", "Paramètres"),
            ("zh", "光谱", "打开", "设置"),
            ("de", "Spektren", "Öffnen", "Einstellungen"),
            ("ru", "Спектры", "Открыть", "Настройки"),
        ],
    )
    def test_startup_restores_persisted_language(
        self, qapp, monkeypatch, lang, tab0, action_open, action_settings
    ):
        monkeypatch.setattr(main_mod.Main, "_prewarm_ai_chat", lambda self: None)
        MSettings().set_language(lang)
        win = main_mod.Main()
        try:
            assert get_language() == lang
            assert win.tabWidget.tabText(0) == tab0
            assert win.menu_bar.actionOpen.text() == action_open
            assert win.menu_bar.actionSettings.text() == action_settings
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
            assert "Respond to the user in" not in en_prompt

            for lang, send_txt, placeholder_txt, custom_txt, hint in [
                ("fr", "Envoyer", "Nouvelle conversation", "Personnalisé", "Respond to the user in French (français)"),
                ("zh", "发送", "新建对话", "自定义", "Respond to the user in Simplified Chinese (简体中文)"),
                ("de", "Senden", "Neue Unterhaltung", "Benutzerdefiniert", "Respond to the user in German (Deutsch)"),
                ("ru", "Отправить", "Новый диалог", "Пользовательский", "Respond to the user in Russian (Русский)"),
            ]:
                set_language(lang, persist=False)
                assert panel.btn_send.text() == send_txt
                assert panel.edit_title.placeholderText() == placeholder_txt
                assert panel.cbb_provider.currentData() == "Custom"
                assert panel.cbb_provider.currentText() == custom_txt
                lang_prompt = panel.vm._build_system_prompt()
                assert lang_prompt.startswith(en_prompt)
                assert hint in lang_prompt

            set_language("en", persist=False)
            assert panel.btn_send.text() == "Send"
            assert panel.cbb_provider.currentData() == "Custom"
        finally:
            panel.close()
            panel.deleteLater()
            qapp.processEvents()


class TestCatalogAndSourceAudit:
    @pytest.mark.parametrize("lang", ["fr", "zh", "de", "ru"])
    def test_catalog_complete_and_valid(self, lang):
        i18n_dir = Path(main_mod.__file__).resolve().parent / "resources" / "i18n"
        ref = json.loads((i18n_dir / "fr.json").read_text(encoding="utf-8"))
        report = validate_catalog(
            lang,
            required_strings=ref["strings"].keys(),
            required_plurals=ref["plurals"].keys(),
        )
        assert report.ok, (
            f"[{lang}] Catalog validation failed: missing_strings={report.missing_strings}, "
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
