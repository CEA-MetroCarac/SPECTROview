# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'gui.ui'
##
## Created by: Qt User Interface Compiler version 6.6.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLayout, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMdiArea, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSpacerItem, QSpinBox, QSplitter, QTabWidget,
    QToolBar, QToolButton, QVBoxLayout, QWidget)
import resources_rc

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        if not mainWindow.objectName():
            mainWindow.setObjectName(u"mainWindow")
        mainWindow.resize(1553, 1019)
        self.actionabout = QAction(mainWindow)
        self.actionabout.setObjectName(u"actionabout")
        self.actionOpen_dataframe_Excel = QAction(mainWindow)
        self.actionOpen_dataframe_Excel.setObjectName(u"actionOpen_dataframe_Excel")
        self.actionOpen_dataframe_CSV = QAction(mainWindow)
        self.actionOpen_dataframe_CSV.setObjectName(u"actionOpen_dataframe_CSV")
        self.actionOpen_saved_work_s = QAction(mainWindow)
        self.actionOpen_saved_work_s.setObjectName(u"actionOpen_saved_work_s")
        self.actionOpen_a_recipie = QAction(mainWindow)
        self.actionOpen_a_recipie.setObjectName(u"actionOpen_a_recipie")
        self.actionSave_all_graph_PNG = QAction(mainWindow)
        self.actionSave_all_graph_PNG.setObjectName(u"actionSave_all_graph_PNG")
        self.actionSave_all_graphs_to_pptx = QAction(mainWindow)
        self.actionSave_all_graphs_to_pptx.setObjectName(u"actionSave_all_graphs_to_pptx")
        self.open_df = QAction(mainWindow)
        self.open_df.setObjectName(u"open_df")
        self.actionHelps = QAction(mainWindow)
        self.actionHelps.setObjectName(u"actionHelps")
        icon = QIcon()
        icon.addFile(u":/icon/iconpack/manual.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionHelps.setIcon(icon)
        self.actionDarkMode = QAction(mainWindow)
        self.actionDarkMode.setObjectName(u"actionDarkMode")
        icon1 = QIcon()
        icon1.addFile(u":/icon/iconpack/dark.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionDarkMode.setIcon(icon1)
        self.actionLightMode = QAction(mainWindow)
        self.actionLightMode.setObjectName(u"actionLightMode")
        self.actionLightMode.setCheckable(False)
        self.actionLightMode.setChecked(False)
        icon2 = QIcon()
        icon2.addFile(u":/icon/iconpack/light-mode.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.actionLightMode.setIcon(icon2)
        self.actionAbout = QAction(mainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        icon3 = QIcon()
        icon3.addFile(u":/icon/iconpack/about.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionAbout.setIcon(icon3)
        self.actionOpen_wafer = QAction(mainWindow)
        self.actionOpen_wafer.setObjectName(u"actionOpen_wafer")
        self.action_reload = QAction(mainWindow)
        self.action_reload.setObjectName(u"action_reload")
        icon4 = QIcon()
        icon4.addFile(u":/icon/iconpack/icons8-documents-folder-96.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_reload.setIcon(icon4)
        self.actionOpen_spectra = QAction(mainWindow)
        self.actionOpen_spectra.setObjectName(u"actionOpen_spectra")
        self.actionOpen_dfs = QAction(mainWindow)
        self.actionOpen_dfs.setObjectName(u"actionOpen_dfs")
        icon5 = QIcon()
        icon5.addFile(u":/icon/iconpack/view.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionOpen_dfs.setIcon(icon5)
        self.actionOpen = QAction(mainWindow)
        self.actionOpen.setObjectName(u"actionOpen")
        icon6 = QIcon()
        icon6.addFile(u":/icon/iconpack/icons8-folder-96.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionOpen.setIcon(icon6)
        self.actionOpen_2 = QAction(mainWindow)
        self.actionOpen_2.setObjectName(u"actionOpen_2")
        self.actionOpen_2.setIcon(icon6)
        self.actionSave = QAction(mainWindow)
        self.actionSave.setObjectName(u"actionSave")
        icon7 = QIcon()
        icon7.addFile(u":/icon/iconpack/save.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionSave.setIcon(icon7)
        self.actionClear_WS = QAction(mainWindow)
        self.actionClear_WS.setObjectName(u"actionClear_WS")
        self.actionThem = QAction(mainWindow)
        self.actionThem.setObjectName(u"actionThem")
        icon8 = QIcon()
        icon8.addFile(u":/icon/iconpack/dark-light.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionThem.setIcon(icon8)
        self.actionClear_env = QAction(mainWindow)
        self.actionClear_env.setObjectName(u"actionClear_env")
        icon9 = QIcon()
        icon9.addFile(u":/icon/iconpack/clear.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionClear_env.setIcon(icon9)
        self.actionLogo = QAction(mainWindow)
        self.actionLogo.setObjectName(u"actionLogo")
        icon10 = QIcon()
        icon10.addFile(u":/icon/logo.png", QSize(), QIcon.Normal, QIcon.Off)
        self.actionLogo.setIcon(icon10)
        self.centralwidget = QWidget(mainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setEnabled(True)
        self.centralwidget.setBaseSize(QSize(0, 0))
        self.verticalLayout_15 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_15.setSpacing(0)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_15.setContentsMargins(5, 5, 5, 5)
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setEnabled(True)
        self.tabWidget.setMinimumSize(QSize(1200, 900))
        self.tabWidget.setMaximumSize(QSize(2560, 1440))
        self.tab_spectra = QWidget()
        self.tab_spectra.setObjectName(u"tab_spectra")
        self.horizontalLayout_131 = QHBoxLayout(self.tab_spectra)
        self.horizontalLayout_131.setObjectName(u"horizontalLayout_131")
        self.horizontalLayout_131.setContentsMargins(5, 5, 5, 5)
        self.splitter_3 = QSplitter(self.tab_spectra)
        self.splitter_3.setObjectName(u"splitter_3")
        self.splitter_3.setOrientation(Qt.Vertical)
        self.splitter_3.setHandleWidth(10)
        self.upper_frame_3 = QFrame(self.splitter_3)
        self.upper_frame_3.setObjectName(u"upper_frame_3")
        self.horizontalLayout_105 = QHBoxLayout(self.upper_frame_3)
        self.horizontalLayout_105.setObjectName(u"horizontalLayout_105")
        self.horizontalLayout_105.setContentsMargins(3, 0, 3, 3)
        self.Upper_zone_3 = QHBoxLayout()
        self.Upper_zone_3.setSpacing(0)
        self.Upper_zone_3.setObjectName(u"Upper_zone_3")
        self.verticalLayout_62 = QVBoxLayout()
        self.verticalLayout_62.setObjectName(u"verticalLayout_62")
        self.verticalLayout_62.setContentsMargins(0, -1, 10, -1)
        self.spectre_view_frame_3 = QFrame(self.upper_frame_3)
        self.spectre_view_frame_3.setObjectName(u"spectre_view_frame_3")
        self.spectre_view_frame_3.setFrameShape(QFrame.StyledPanel)
        self.spectre_view_frame_3.setFrameShadow(QFrame.Raised)
        self.verticalLayout_63 = QVBoxLayout(self.spectre_view_frame_3)
        self.verticalLayout_63.setObjectName(u"verticalLayout_63")
        self.verticalLayout_63.setContentsMargins(0, 0, 0, 0)
        self.QVBoxlayout_2 = QVBoxLayout()
        self.QVBoxlayout_2.setSpacing(6)
        self.QVBoxlayout_2.setObjectName(u"QVBoxlayout_2")

        self.verticalLayout_63.addLayout(self.QVBoxlayout_2)


        self.verticalLayout_62.addWidget(self.spectre_view_frame_3)

        self.bottom_frame_3 = QHBoxLayout()
        self.bottom_frame_3.setSpacing(10)
        self.bottom_frame_3.setObjectName(u"bottom_frame_3")
        self.bottom_frame_3.setSizeConstraint(QLayout.SetMaximumSize)
        self.bottom_frame_3.setContentsMargins(2, 2, 2, 2)
        self.toolbar_frame_3 = QHBoxLayout()
        self.toolbar_frame_3.setObjectName(u"toolbar_frame_3")

        self.bottom_frame_3.addLayout(self.toolbar_frame_3)

        self.horizontalSpacer_24 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottom_frame_3.addItem(self.horizontalSpacer_24)

        self.rdbtn_baseline_2 = QRadioButton(self.upper_frame_3)
        self.rdbtn_baseline_2.setObjectName(u"rdbtn_baseline_2")
        self.rdbtn_baseline_2.setChecked(True)

        self.bottom_frame_3.addWidget(self.rdbtn_baseline_2)

        self.rdbtn_peak_2 = QRadioButton(self.upper_frame_3)
        self.rdbtn_peak_2.setObjectName(u"rdbtn_peak_2")
        self.rdbtn_peak_2.setChecked(False)

        self.bottom_frame_3.addWidget(self.rdbtn_peak_2)

        self.horizontalSpacer_16 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottom_frame_3.addItem(self.horizontalSpacer_16)

        self.rsquared_2 = QLabel(self.upper_frame_3)
        self.rsquared_2.setObjectName(u"rsquared_2")

        self.bottom_frame_3.addWidget(self.rsquared_2)

        self.btn_copy_fig_3 = QPushButton(self.upper_frame_3)
        self.btn_copy_fig_3.setObjectName(u"btn_copy_fig_3")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_copy_fig_3.sizePolicy().hasHeightForWidth())
        self.btn_copy_fig_3.setSizePolicy(sizePolicy)
        self.btn_copy_fig_3.setMinimumSize(QSize(0, 0))
        self.btn_copy_fig_3.setMaximumSize(QSize(16777215, 16777215))
        icon11 = QIcon()
        icon11.addFile(u":/icon/iconpack/copy.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_copy_fig_3.setIcon(icon11)
        self.btn_copy_fig_3.setIconSize(QSize(24, 24))

        self.bottom_frame_3.addWidget(self.btn_copy_fig_3)

        self.label_79 = QLabel(self.upper_frame_3)
        self.label_79.setObjectName(u"label_79")

        self.bottom_frame_3.addWidget(self.label_79)

        self.sb_dpi_spectra_2 = QSpinBox(self.upper_frame_3)
        self.sb_dpi_spectra_2.setObjectName(u"sb_dpi_spectra_2")
        self.sb_dpi_spectra_2.setMinimum(100)
        self.sb_dpi_spectra_2.setMaximum(200)
        self.sb_dpi_spectra_2.setSingleStep(10)

        self.bottom_frame_3.addWidget(self.sb_dpi_spectra_2)

        self.horizontalSpacer_25 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottom_frame_3.addItem(self.horizontalSpacer_25)

        self.bottom_frame_3.setStretch(0, 50)
        self.bottom_frame_3.setStretch(1, 25)
        self.bottom_frame_3.setStretch(9, 2)

        self.verticalLayout_62.addLayout(self.bottom_frame_3)

        self.verticalLayout_62.setStretch(0, 75)
        self.verticalLayout_62.setStretch(1, 25)

        self.Upper_zone_3.addLayout(self.verticalLayout_62)

        self.widget_9 = QWidget(self.upper_frame_3)
        self.widget_9.setObjectName(u"widget_9")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.widget_9.sizePolicy().hasHeightForWidth())
        self.widget_9.setSizePolicy(sizePolicy1)
        self.widget_9.setMinimumSize(QSize(300, 0))
        self.verticalLayout_64 = QVBoxLayout(self.widget_9)
        self.verticalLayout_64.setObjectName(u"verticalLayout_64")
        self.verticalLayout_64.setContentsMargins(2, 0, 2, 0)
        self.scrollArea_7 = QScrollArea(self.widget_9)
        self.scrollArea_7.setObjectName(u"scrollArea_7")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.scrollArea_7.sizePolicy().hasHeightForWidth())
        self.scrollArea_7.setSizePolicy(sizePolicy2)
        self.scrollArea_7.setMinimumSize(QSize(250, 450))
        self.scrollArea_7.setMaximumSize(QSize(350, 16777215))
        self.scrollArea_7.setWidgetResizable(True)
        self.scrollAreaWidgetContents_7 = QWidget()
        self.scrollAreaWidgetContents_7.setObjectName(u"scrollAreaWidgetContents_7")
        self.scrollAreaWidgetContents_7.setGeometry(QRect(0, 0, 337, 448))
        self.verticalLayout_74 = QVBoxLayout(self.scrollAreaWidgetContents_7)
        self.verticalLayout_74.setObjectName(u"verticalLayout_74")
        self.toolButton = QToolButton(self.scrollAreaWidgetContents_7)
        self.toolButton.setObjectName(u"toolButton")

        self.verticalLayout_74.addWidget(self.toolButton)

        self.verticalSpacer_12 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_74.addItem(self.verticalSpacer_12)

        self.view_options_box_2 = QGroupBox(self.scrollAreaWidgetContents_7)
        self.view_options_box_2.setObjectName(u"view_options_box_2")
        self.view_options_box_2.setMaximumSize(QSize(320, 16777215))
        self.gridLayout_7 = QGridLayout(self.view_options_box_2)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.cb_residual_3 = QCheckBox(self.view_options_box_2)
        self.cb_residual_3.setObjectName(u"cb_residual_3")
        self.cb_residual_3.setChecked(False)

        self.gridLayout_7.addWidget(self.cb_residual_3, 1, 1, 1, 1)

        self.cb_filled_3 = QCheckBox(self.view_options_box_2)
        self.cb_filled_3.setObjectName(u"cb_filled_3")
        self.cb_filled_3.setChecked(True)

        self.gridLayout_7.addWidget(self.cb_filled_3, 0, 2, 1, 1)

        self.cb_bestfit_3 = QCheckBox(self.view_options_box_2)
        self.cb_bestfit_3.setObjectName(u"cb_bestfit_3")
        self.cb_bestfit_3.setChecked(True)

        self.gridLayout_7.addWidget(self.cb_bestfit_3, 0, 1, 1, 1)

        self.cb_legend_3 = QCheckBox(self.view_options_box_2)
        self.cb_legend_3.setObjectName(u"cb_legend_3")
        self.cb_legend_3.setEnabled(True)
        self.cb_legend_3.setChecked(False)

        self.gridLayout_7.addWidget(self.cb_legend_3, 0, 0, 1, 1)

        self.cb_raw_3 = QCheckBox(self.view_options_box_2)
        self.cb_raw_3.setObjectName(u"cb_raw_3")
        self.cb_raw_3.setChecked(False)

        self.gridLayout_7.addWidget(self.cb_raw_3, 1, 0, 1, 1)

        self.cb_colors_3 = QCheckBox(self.view_options_box_2)
        self.cb_colors_3.setObjectName(u"cb_colors_3")
        self.cb_colors_3.setChecked(True)

        self.gridLayout_7.addWidget(self.cb_colors_3, 1, 2, 1, 1)

        self.cb_peaks_3 = QCheckBox(self.view_options_box_2)
        self.cb_peaks_3.setObjectName(u"cb_peaks_3")
        self.cb_peaks_3.setChecked(False)

        self.gridLayout_7.addWidget(self.cb_peaks_3, 0, 3, 1, 1)

        self.cb_normalize_3 = QCheckBox(self.view_options_box_2)
        self.cb_normalize_3.setObjectName(u"cb_normalize_3")

        self.gridLayout_7.addWidget(self.cb_normalize_3, 1, 3, 1, 1)


        self.verticalLayout_74.addWidget(self.view_options_box_2)

        self.scrollArea_7.setWidget(self.scrollAreaWidgetContents_7)

        self.verticalLayout_64.addWidget(self.scrollArea_7)


        self.Upper_zone_3.addWidget(self.widget_9)

        self.Upper_zone_3.setStretch(0, 75)

        self.horizontalLayout_105.addLayout(self.Upper_zone_3)

        self.splitter_3.addWidget(self.upper_frame_3)
        self.bottom_widget_4 = QWidget(self.splitter_3)
        self.bottom_widget_4.setObjectName(u"bottom_widget_4")
        self.verticalLayout_66 = QVBoxLayout(self.bottom_widget_4)
        self.verticalLayout_66.setObjectName(u"verticalLayout_66")
        self.verticalLayout_66.setContentsMargins(3, 3, 3, 0)
        self.tabWidget_3 = QTabWidget(self.bottom_widget_4)
        self.tabWidget_3.setObjectName(u"tabWidget_3")
        self.tabWidget_3.setEnabled(True)
        self.fit_model_editor_3 = QWidget()
        self.fit_model_editor_3.setObjectName(u"fit_model_editor_3")
        self.fit_model_editor_3.setEnabled(True)
        self.verticalLayout_46 = QVBoxLayout(self.fit_model_editor_3)
        self.verticalLayout_46.setSpacing(6)
        self.verticalLayout_46.setObjectName(u"verticalLayout_46")
        self.verticalLayout_46.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_72 = QHBoxLayout()
        self.horizontalLayout_72.setSpacing(5)
        self.horizontalLayout_72.setObjectName(u"horizontalLayout_72")
        self.horizontalLayout_72.setContentsMargins(-1, 5, 5, 5)
        self.widget_18 = QWidget(self.fit_model_editor_3)
        self.widget_18.setObjectName(u"widget_18")
        self.horizontalLayout_73 = QHBoxLayout(self.widget_18)
        self.horizontalLayout_73.setSpacing(6)
        self.horizontalLayout_73.setObjectName(u"horizontalLayout_73")
        self.horizontalLayout_73.setContentsMargins(2, 2, 2, 2)
        self.scrollArea_4 = QScrollArea(self.widget_18)
        self.scrollArea_4.setObjectName(u"scrollArea_4")
        self.scrollArea_4.setMinimumSize(QSize(430, 100))
        self.scrollArea_4.setMaximumSize(QSize(430, 16777215))
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollAreaWidgetContents_4 = QWidget()
        self.scrollAreaWidgetContents_4.setObjectName(u"scrollAreaWidgetContents_4")
        self.scrollAreaWidgetContents_4.setGeometry(QRect(0, 0, 428, 380))
        self.verticalLayout_39 = QVBoxLayout(self.scrollAreaWidgetContents_4)
        self.verticalLayout_39.setSpacing(10)
        self.verticalLayout_39.setObjectName(u"verticalLayout_39")
        self.verticalLayout_39.setContentsMargins(10, 10, 10, 10)
        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.btn_cosmis_ray_3 = QPushButton(self.scrollAreaWidgetContents_4)
        self.btn_cosmis_ray_3.setObjectName(u"btn_cosmis_ray_3")
        self.btn_cosmis_ray_3.setMinimumSize(QSize(80, 0))
        self.btn_cosmis_ray_3.setMaximumSize(QSize(150, 16777215))

        self.horizontalLayout_17.addWidget(self.btn_cosmis_ray_3)

        self.horizontalSpacer_57 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_57)

        self.label_22 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_22.setObjectName(u"label_22")

        self.horizontalLayout_17.addWidget(self.label_22)

        self.cbb_xaxis_unit = QComboBox(self.scrollAreaWidgetContents_4)
        self.cbb_xaxis_unit.setObjectName(u"cbb_xaxis_unit")

        self.horizontalLayout_17.addWidget(self.cbb_xaxis_unit)


        self.verticalLayout_39.addLayout(self.horizontalLayout_17)

        self.groupBox_5 = QGroupBox(self.scrollAreaWidgetContents_4)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.verticalLayout_40 = QVBoxLayout(self.groupBox_5)
        self.verticalLayout_40.setSpacing(5)
        self.verticalLayout_40.setObjectName(u"verticalLayout_40")
        self.verticalLayout_40.setContentsMargins(2, 2, 2, 2)
        self.label_65 = QLabel(self.groupBox_5)
        self.label_65.setObjectName(u"label_65")
        font = QFont()
        font.setBold(True)
        self.label_65.setFont(font)

        self.verticalLayout_40.addWidget(self.label_65)

        self.horizontalLayout_74 = QHBoxLayout()
        self.horizontalLayout_74.setSpacing(5)
        self.horizontalLayout_74.setObjectName(u"horizontalLayout_74")
        self.horizontalLayout_74.setContentsMargins(2, 2, 2, 2)
        self.label_66 = QLabel(self.groupBox_5)
        self.label_66.setObjectName(u"label_66")

        self.horizontalLayout_74.addWidget(self.label_66)

        self.horizontalSpacer_34 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_74.addItem(self.horizontalSpacer_34)

        self.range_min_2 = QLineEdit(self.groupBox_5)
        self.range_min_2.setObjectName(u"range_min_2")

        self.horizontalLayout_74.addWidget(self.range_min_2)

        self.label_67 = QLabel(self.groupBox_5)
        self.label_67.setObjectName(u"label_67")

        self.horizontalLayout_74.addWidget(self.label_67)

        self.range_max_2 = QLineEdit(self.groupBox_5)
        self.range_max_2.setObjectName(u"range_max_2")

        self.horizontalLayout_74.addWidget(self.range_max_2)

        self.range_apply_2 = QPushButton(self.groupBox_5)
        self.range_apply_2.setObjectName(u"range_apply_2")

        self.horizontalLayout_74.addWidget(self.range_apply_2)


        self.verticalLayout_40.addLayout(self.horizontalLayout_74)


        self.verticalLayout_39.addWidget(self.groupBox_5)

        self.label_68 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_68.setObjectName(u"label_68")

        self.verticalLayout_39.addWidget(self.label_68)

        self.baseline_2 = QGroupBox(self.scrollAreaWidgetContents_4)
        self.baseline_2.setObjectName(u"baseline_2")
        self.verticalLayout_41 = QVBoxLayout(self.baseline_2)
        self.verticalLayout_41.setSpacing(5)
        self.verticalLayout_41.setObjectName(u"verticalLayout_41")
        self.verticalLayout_41.setContentsMargins(2, 2, 2, 2)
        self.label_69 = QLabel(self.baseline_2)
        self.label_69.setObjectName(u"label_69")
        self.label_69.setFont(font)

        self.verticalLayout_41.addWidget(self.label_69)

        self.horizontalLayout_75 = QHBoxLayout()
        self.horizontalLayout_75.setSpacing(5)
        self.horizontalLayout_75.setObjectName(u"horizontalLayout_75")
        self.horizontalLayout_75.setContentsMargins(2, 2, 2, 2)
        self.rbtn_linear_2 = QRadioButton(self.baseline_2)
        self.rbtn_linear_2.setObjectName(u"rbtn_linear_2")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.rbtn_linear_2.sizePolicy().hasHeightForWidth())
        self.rbtn_linear_2.setSizePolicy(sizePolicy3)
        self.rbtn_linear_2.setChecked(True)

        self.horizontalLayout_75.addWidget(self.rbtn_linear_2)

        self.rbtn_polynomial_2 = QRadioButton(self.baseline_2)
        self.rbtn_polynomial_2.setObjectName(u"rbtn_polynomial_2")
        sizePolicy3.setHeightForWidth(self.rbtn_polynomial_2.sizePolicy().hasHeightForWidth())
        self.rbtn_polynomial_2.setSizePolicy(sizePolicy3)

        self.horizontalLayout_75.addWidget(self.rbtn_polynomial_2)

        self.degre_2 = QSpinBox(self.baseline_2)
        self.degre_2.setObjectName(u"degre_2")
        self.degre_2.setMinimum(1)

        self.horizontalLayout_75.addWidget(self.degre_2)

        self.cb_attached_2 = QCheckBox(self.baseline_2)
        self.cb_attached_2.setObjectName(u"cb_attached_2")
        sizePolicy3.setHeightForWidth(self.cb_attached_2.sizePolicy().hasHeightForWidth())
        self.cb_attached_2.setSizePolicy(sizePolicy3)
        self.cb_attached_2.setChecked(True)

        self.horizontalLayout_75.addWidget(self.cb_attached_2)

        self.horizontalSpacer_36 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_75.addItem(self.horizontalSpacer_36)

        self.label_70 = QLabel(self.baseline_2)
        self.label_70.setObjectName(u"label_70")

        self.horizontalLayout_75.addWidget(self.label_70)

        self.noise_2 = QDoubleSpinBox(self.baseline_2)
        self.noise_2.setObjectName(u"noise_2")
        self.noise_2.setDecimals(0)
        self.noise_2.setValue(5.000000000000000)

        self.horizontalLayout_75.addWidget(self.noise_2)

        self.horizontalLayout_75.setStretch(0, 25)
        self.horizontalLayout_75.setStretch(1, 25)

        self.verticalLayout_41.addLayout(self.horizontalLayout_75)

        self.horizontalLayout_76 = QHBoxLayout()
        self.horizontalLayout_76.setSpacing(5)
        self.horizontalLayout_76.setObjectName(u"horizontalLayout_76")
        self.horizontalLayout_76.setContentsMargins(2, 2, 2, 2)
        self.btn_undo_baseline_2 = QPushButton(self.baseline_2)
        self.btn_undo_baseline_2.setObjectName(u"btn_undo_baseline_2")
        icon12 = QIcon()
        icon12.addFile(u":/icon/iconpack/remove.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_undo_baseline_2.setIcon(icon12)

        self.horizontalLayout_76.addWidget(self.btn_undo_baseline_2)

        self.horizontalSpacer_37 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_76.addItem(self.horizontalSpacer_37)

        self.btn_copy_bl_2 = QPushButton(self.baseline_2)
        self.btn_copy_bl_2.setObjectName(u"btn_copy_bl_2")
        self.btn_copy_bl_2.setIcon(icon11)

        self.horizontalLayout_76.addWidget(self.btn_copy_bl_2)

        self.btn_paste_bl_2 = QPushButton(self.baseline_2)
        self.btn_paste_bl_2.setObjectName(u"btn_paste_bl_2")
        icon13 = QIcon()
        icon13.addFile(u":/icon/iconpack/paste10.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_paste_bl_2.setIcon(icon13)

        self.horizontalLayout_76.addWidget(self.btn_paste_bl_2)

        self.sub_baseline_2 = QPushButton(self.baseline_2)
        self.sub_baseline_2.setObjectName(u"sub_baseline_2")

        self.horizontalLayout_76.addWidget(self.sub_baseline_2)


        self.verticalLayout_41.addLayout(self.horizontalLayout_76)


        self.verticalLayout_39.addWidget(self.baseline_2)

        self.label_71 = QLabel(self.scrollAreaWidgetContents_4)
        self.label_71.setObjectName(u"label_71")

        self.verticalLayout_39.addWidget(self.label_71)

        self.peaks_2 = QGroupBox(self.scrollAreaWidgetContents_4)
        self.peaks_2.setObjectName(u"peaks_2")
        self.verticalLayout_42 = QVBoxLayout(self.peaks_2)
        self.verticalLayout_42.setSpacing(5)
        self.verticalLayout_42.setObjectName(u"verticalLayout_42")
        self.verticalLayout_42.setContentsMargins(2, 2, 2, 2)
        self.label_72 = QLabel(self.peaks_2)
        self.label_72.setObjectName(u"label_72")
        self.label_72.setFont(font)

        self.verticalLayout_42.addWidget(self.label_72)

        self.horizontalLayout_77 = QHBoxLayout()
        self.horizontalLayout_77.setSpacing(5)
        self.horizontalLayout_77.setObjectName(u"horizontalLayout_77")
        self.horizontalLayout_77.setContentsMargins(2, 2, 2, 2)
        self.label_73 = QLabel(self.peaks_2)
        self.label_73.setObjectName(u"label_73")

        self.horizontalLayout_77.addWidget(self.label_73)

        self.horizontalSpacer_38 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_77.addItem(self.horizontalSpacer_38)

        self.cbb_fit_models_2 = QComboBox(self.peaks_2)
        self.cbb_fit_models_2.setObjectName(u"cbb_fit_models_2")

        self.horizontalLayout_77.addWidget(self.cbb_fit_models_2)

        self.clear_peaks_2 = QPushButton(self.peaks_2)
        self.clear_peaks_2.setObjectName(u"clear_peaks_2")
        self.clear_peaks_2.setIcon(icon12)

        self.horizontalLayout_77.addWidget(self.clear_peaks_2)

        self.horizontalLayout_77.setStretch(2, 65)

        self.verticalLayout_42.addLayout(self.horizontalLayout_77)


        self.verticalLayout_39.addWidget(self.peaks_2)

        self.verticalSpacer_15 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_39.addItem(self.verticalSpacer_15)

        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)

        self.horizontalLayout_73.addWidget(self.scrollArea_4)

        self.verticalLayout_43 = QVBoxLayout()
        self.verticalLayout_43.setObjectName(u"verticalLayout_43")
        self.peak_table_2 = QGroupBox(self.widget_18)
        self.peak_table_2.setObjectName(u"peak_table_2")
        self.horizontalLayout_78 = QHBoxLayout(self.peak_table_2)
        self.horizontalLayout_78.setObjectName(u"horizontalLayout_78")
        self.scrollArea_6 = QScrollArea(self.peak_table_2)
        self.scrollArea_6.setObjectName(u"scrollArea_6")
        self.scrollArea_6.setWidgetResizable(True)
        self.scrollAreaWidgetContents_6 = QWidget()
        self.scrollAreaWidgetContents_6.setObjectName(u"scrollAreaWidgetContents_6")
        self.scrollAreaWidgetContents_6.setGeometry(QRect(0, 0, 717, 221))
        self.verticalLayout_44 = QVBoxLayout(self.scrollAreaWidgetContents_6)
        self.verticalLayout_44.setObjectName(u"verticalLayout_44")
        self.verticalLayout_312 = QVBoxLayout()
        self.verticalLayout_312.setObjectName(u"verticalLayout_312")
        self.horizontalLayout_79 = QHBoxLayout()
        self.horizontalLayout_79.setObjectName(u"horizontalLayout_79")
        self.peak_table1_2 = QHBoxLayout()
        self.peak_table1_2.setObjectName(u"peak_table1_2")

        self.horizontalLayout_79.addLayout(self.peak_table1_2)

        self.horizontalLayout_80 = QHBoxLayout()
        self.horizontalLayout_80.setObjectName(u"horizontalLayout_80")

        self.horizontalLayout_79.addLayout(self.horizontalLayout_80)


        self.verticalLayout_312.addLayout(self.horizontalLayout_79)

        self.verticalSpacer_16 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_312.addItem(self.verticalSpacer_16)


        self.verticalLayout_44.addLayout(self.verticalLayout_312)

        self.scrollArea_6.setWidget(self.scrollAreaWidgetContents_6)

        self.horizontalLayout_78.addWidget(self.scrollArea_6)


        self.verticalLayout_43.addWidget(self.peak_table_2)

        self.horizontalLayout_81 = QHBoxLayout()
        self.horizontalLayout_81.setSpacing(5)
        self.horizontalLayout_81.setObjectName(u"horizontalLayout_81")
        self.btn_fit_3 = QPushButton(self.widget_18)
        self.btn_fit_3.setObjectName(u"btn_fit_3")
        self.btn_fit_3.setMinimumSize(QSize(50, 50))
        self.btn_fit_3.setMaximumSize(QSize(50, 50))

        self.horizontalLayout_81.addWidget(self.btn_fit_3)

        self.verticalLayout_45 = QVBoxLayout()
        self.verticalLayout_45.setObjectName(u"verticalLayout_45")
        self.horizontalLayout_82 = QHBoxLayout()
        self.horizontalLayout_82.setSpacing(5)
        self.horizontalLayout_82.setObjectName(u"horizontalLayout_82")
        self.horizontalLayout_82.setContentsMargins(5, 2, 5, 2)
        self.btn_copy_fit_model_2 = QPushButton(self.widget_18)
        self.btn_copy_fit_model_2.setObjectName(u"btn_copy_fit_model_2")
        self.btn_copy_fit_model_2.setIcon(icon11)

        self.horizontalLayout_82.addWidget(self.btn_copy_fit_model_2)

        self.lbl_copied_fit_model_2 = QLabel(self.widget_18)
        self.lbl_copied_fit_model_2.setObjectName(u"lbl_copied_fit_model_2")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.lbl_copied_fit_model_2.sizePolicy().hasHeightForWidth())
        self.lbl_copied_fit_model_2.setSizePolicy(sizePolicy4)
        self.lbl_copied_fit_model_2.setMinimumSize(QSize(50, 0))

        self.horizontalLayout_82.addWidget(self.lbl_copied_fit_model_2)

        self.btn_paste_fit_model_2 = QPushButton(self.widget_18)
        self.btn_paste_fit_model_2.setObjectName(u"btn_paste_fit_model_2")
        self.btn_paste_fit_model_2.setMinimumSize(QSize(0, 0))
        self.btn_paste_fit_model_2.setMaximumSize(QSize(16777215, 40))
        self.btn_paste_fit_model_2.setIcon(icon13)

        self.horizontalLayout_82.addWidget(self.btn_paste_fit_model_2)

        self.horizontalSpacer_51 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_82.addItem(self.horizontalSpacer_51)

        self.save_model_2 = QPushButton(self.widget_18)
        self.save_model_2.setObjectName(u"save_model_2")
        icon14 = QIcon()
        icon14.addFile(u":/icon/iconpack/save11.png", QSize(), QIcon.Normal, QIcon.Off)
        self.save_model_2.setIcon(icon14)

        self.horizontalLayout_82.addWidget(self.save_model_2)

        self.horizontalSpacer_39 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_82.addItem(self.horizontalSpacer_39)

        self.cb_limits_2 = QCheckBox(self.widget_18)
        self.cb_limits_2.setObjectName(u"cb_limits_2")

        self.horizontalLayout_82.addWidget(self.cb_limits_2)

        self.cb_expr_2 = QCheckBox(self.widget_18)
        self.cb_expr_2.setObjectName(u"cb_expr_2")

        self.horizontalLayout_82.addWidget(self.cb_expr_2)


        self.verticalLayout_45.addLayout(self.horizontalLayout_82)

        self.horizontalLayout_106 = QHBoxLayout()
        self.horizontalLayout_106.setSpacing(5)
        self.horizontalLayout_106.setObjectName(u"horizontalLayout_106")
        self.horizontalLayout_106.setContentsMargins(5, 2, 5, 2)
        self.label_81 = QLabel(self.widget_18)
        self.label_81.setObjectName(u"label_81")

        self.horizontalLayout_106.addWidget(self.label_81)

        self.cbb_fit_model_list_3 = QComboBox(self.widget_18)
        self.cbb_fit_model_list_3.setObjectName(u"cbb_fit_model_list_3")
        self.cbb_fit_model_list_3.setMinimumSize(QSize(400, 0))
        self.cbb_fit_model_list_3.setMaximumSize(QSize(400, 16777215))

        self.horizontalLayout_106.addWidget(self.cbb_fit_model_list_3)

        self.btn_apply_model_3 = QPushButton(self.widget_18)
        self.btn_apply_model_3.setObjectName(u"btn_apply_model_3")

        self.horizontalLayout_106.addWidget(self.btn_apply_model_3)

        self.horizontalSpacer_42 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_106.addItem(self.horizontalSpacer_42)

        self.horizontalSpacer_26 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_106.addItem(self.horizontalSpacer_26)

        self.btn_load_model_3 = QPushButton(self.widget_18)
        self.btn_load_model_3.setObjectName(u"btn_load_model_3")

        self.horizontalLayout_106.addWidget(self.btn_load_model_3)


        self.verticalLayout_45.addLayout(self.horizontalLayout_106)


        self.horizontalLayout_81.addLayout(self.verticalLayout_45)


        self.verticalLayout_43.addLayout(self.horizontalLayout_81)

        self.verticalLayout_43.setStretch(0, 85)
        self.verticalLayout_43.setStretch(1, 15)

        self.horizontalLayout_73.addLayout(self.verticalLayout_43)

        self.horizontalLayout_73.setStretch(0, 50)
        self.horizontalLayout_73.setStretch(1, 50)

        self.horizontalLayout_72.addWidget(self.widget_18)


        self.verticalLayout_46.addLayout(self.horizontalLayout_72)

        self.tabWidget_3.addTab(self.fit_model_editor_3, "")
        self.collect_fit_data_2 = QWidget()
        self.collect_fit_data_2.setObjectName(u"collect_fit_data_2")
        self.verticalLayout_47 = QVBoxLayout(self.collect_fit_data_2)
        self.verticalLayout_47.setSpacing(6)
        self.verticalLayout_47.setObjectName(u"verticalLayout_47")
        self.verticalLayout_47.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_94 = QHBoxLayout()
        self.horizontalLayout_94.setSpacing(5)
        self.horizontalLayout_94.setObjectName(u"horizontalLayout_94")
        self.horizontalLayout_94.setContentsMargins(5, 5, 5, 5)
        self.scrollArea_11 = QScrollArea(self.collect_fit_data_2)
        self.scrollArea_11.setObjectName(u"scrollArea_11")
        sizePolicy2.setHeightForWidth(self.scrollArea_11.sizePolicy().hasHeightForWidth())
        self.scrollArea_11.setSizePolicy(sizePolicy2)
        self.scrollArea_11.setMinimumSize(QSize(430, 100))
        self.scrollArea_11.setMaximumSize(QSize(430, 16777215))
        self.scrollArea_11.setWidgetResizable(True)
        self.scrollAreaWidgetContents_11 = QWidget()
        self.scrollAreaWidgetContents_11.setObjectName(u"scrollAreaWidgetContents_11")
        self.scrollAreaWidgetContents_11.setGeometry(QRect(0, 0, 283, 217))
        self.verticalLayout_81 = QVBoxLayout(self.scrollAreaWidgetContents_11)
        self.verticalLayout_81.setSpacing(10)
        self.verticalLayout_81.setObjectName(u"verticalLayout_81")
        self.verticalLayout_81.setContentsMargins(10, 10, 10, 10)
        self.btn_collect_results_3 = QPushButton(self.scrollAreaWidgetContents_11)
        self.btn_collect_results_3.setObjectName(u"btn_collect_results_3")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.btn_collect_results_3.sizePolicy().hasHeightForWidth())
        self.btn_collect_results_3.setSizePolicy(sizePolicy5)
        self.btn_collect_results_3.setMinimumSize(QSize(140, 40))
        self.btn_collect_results_3.setMaximumSize(QSize(140, 40))
        self.btn_collect_results_3.setFont(font)
        icon15 = QIcon()
        icon15.addFile(u":/icon/iconpack/collect.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_collect_results_3.setIcon(icon15)
        self.btn_collect_results_3.setIconSize(QSize(16, 22))

        self.verticalLayout_81.addWidget(self.btn_collect_results_3)

        self.label_83 = QLabel(self.scrollAreaWidgetContents_11)
        self.label_83.setObjectName(u"label_83")

        self.verticalLayout_81.addWidget(self.label_83)

        self.horizontalLayout_95 = QHBoxLayout()
        self.horizontalLayout_95.setObjectName(u"horizontalLayout_95")
        self.btn_split_fname = QPushButton(self.scrollAreaWidgetContents_11)
        self.btn_split_fname.setObjectName(u"btn_split_fname")
        sizePolicy5.setHeightForWidth(self.btn_split_fname.sizePolicy().hasHeightForWidth())
        self.btn_split_fname.setSizePolicy(sizePolicy5)
        self.btn_split_fname.setMinimumSize(QSize(40, 0))
        self.btn_split_fname.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_95.addWidget(self.btn_split_fname)

        self.cbb_split_fname = QComboBox(self.scrollAreaWidgetContents_11)
        self.cbb_split_fname.setObjectName(u"cbb_split_fname")
        sizePolicy5.setHeightForWidth(self.cbb_split_fname.sizePolicy().hasHeightForWidth())
        self.cbb_split_fname.setSizePolicy(sizePolicy5)
        self.cbb_split_fname.setMinimumSize(QSize(120, 0))
        self.cbb_split_fname.setMaximumSize(QSize(120, 16777215))

        self.horizontalLayout_95.addWidget(self.cbb_split_fname)

        self.ent_col_name = QLineEdit(self.scrollAreaWidgetContents_11)
        self.ent_col_name.setObjectName(u"ent_col_name")

        self.horizontalLayout_95.addWidget(self.ent_col_name)

        self.btn_add_col = QPushButton(self.scrollAreaWidgetContents_11)
        self.btn_add_col.setObjectName(u"btn_add_col")
        self.btn_add_col.setMinimumSize(QSize(60, 0))
        self.btn_add_col.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout_95.addWidget(self.btn_add_col)

        self.horizontalLayout_95.setStretch(2, 40)
        self.horizontalLayout_95.setStretch(3, 20)

        self.verticalLayout_81.addLayout(self.horizontalLayout_95)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.ent_send_df_to_viz = QLineEdit(self.scrollAreaWidgetContents_11)
        self.ent_send_df_to_viz.setObjectName(u"ent_send_df_to_viz")

        self.horizontalLayout_21.addWidget(self.ent_send_df_to_viz)

        self.btn_send_to_viz = QPushButton(self.scrollAreaWidgetContents_11)
        self.btn_send_to_viz.setObjectName(u"btn_send_to_viz")

        self.horizontalLayout_21.addWidget(self.btn_send_to_viz)


        self.verticalLayout_81.addLayout(self.horizontalLayout_21)

        self.verticalSpacer_18 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_81.addItem(self.verticalSpacer_18)

        self.horizontalLayout_139 = QHBoxLayout()
        self.horizontalLayout_139.setObjectName(u"horizontalLayout_139")
        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_139.addItem(self.horizontalSpacer_10)

        self.btn_view_df_5 = QPushButton(self.scrollAreaWidgetContents_11)
        self.btn_view_df_5.setObjectName(u"btn_view_df_5")
        sizePolicy5.setHeightForWidth(self.btn_view_df_5.sizePolicy().hasHeightForWidth())
        self.btn_view_df_5.setSizePolicy(sizePolicy5)
        self.btn_view_df_5.setMinimumSize(QSize(30, 0))
        self.btn_view_df_5.setMaximumSize(QSize(30, 16777215))
        icon16 = QIcon()
        icon16.addFile(u":/icon/iconpack/view11.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_view_df_5.setIcon(icon16)
        self.btn_view_df_5.setIconSize(QSize(22, 22))

        self.horizontalLayout_139.addWidget(self.btn_view_df_5)

        self.btn_save_fit_results_3 = QPushButton(self.scrollAreaWidgetContents_11)
        self.btn_save_fit_results_3.setObjectName(u"btn_save_fit_results_3")
        sizePolicy5.setHeightForWidth(self.btn_save_fit_results_3.sizePolicy().hasHeightForWidth())
        self.btn_save_fit_results_3.setSizePolicy(sizePolicy5)
        self.btn_save_fit_results_3.setMinimumSize(QSize(30, 0))
        self.btn_save_fit_results_3.setMaximumSize(QSize(30, 16777215))
        icon17 = QIcon()
        icon17.addFile(u":/icon/iconpack/save12.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_save_fit_results_3.setIcon(icon17)
        self.btn_save_fit_results_3.setIconSize(QSize(22, 22))

        self.horizontalLayout_139.addWidget(self.btn_save_fit_results_3)

        self.btn_open_fit_results_3 = QPushButton(self.scrollAreaWidgetContents_11)
        self.btn_open_fit_results_3.setObjectName(u"btn_open_fit_results_3")
        sizePolicy3.setHeightForWidth(self.btn_open_fit_results_3.sizePolicy().hasHeightForWidth())
        self.btn_open_fit_results_3.setSizePolicy(sizePolicy3)
        self.btn_open_fit_results_3.setMaximumSize(QSize(30, 16777215))
        icon18 = QIcon()
        icon18.addFile(u":/icon/iconpack/opened-folder.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_open_fit_results_3.setIcon(icon18)
        self.btn_open_fit_results_3.setIconSize(QSize(22, 22))

        self.horizontalLayout_139.addWidget(self.btn_open_fit_results_3)


        self.verticalLayout_81.addLayout(self.horizontalLayout_139)

        self.scrollArea_11.setWidget(self.scrollAreaWidgetContents_11)

        self.horizontalLayout_94.addWidget(self.scrollArea_11)

        self.verticalLayout_55 = QVBoxLayout()
        self.verticalLayout_55.setObjectName(u"verticalLayout_55")
        self.verticalLayout_55.setContentsMargins(15, -1, -1, -1)
        self.layout_df_table2 = QVBoxLayout()
        self.layout_df_table2.setObjectName(u"layout_df_table2")

        self.verticalLayout_55.addLayout(self.layout_df_table2)

        self.groupBox_6 = QGroupBox(self.collect_fit_data_2)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.horizontalLayout_98 = QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_98.setSpacing(9)
        self.horizontalLayout_98.setObjectName(u"horizontalLayout_98")
        self.horizontalLayout_98.setContentsMargins(3, 3, 3, 3)
        self.horizontalSpacer_46 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_98.addItem(self.horizontalSpacer_46)


        self.verticalLayout_55.addWidget(self.groupBox_6)


        self.horizontalLayout_94.addLayout(self.verticalLayout_55)


        self.verticalLayout_47.addLayout(self.horizontalLayout_94)

        self.tabWidget_3.addTab(self.collect_fit_data_2, "")
        self.fit_settings_3 = QWidget()
        self.fit_settings_3.setObjectName(u"fit_settings_3")
        self.fit_settings_3.setEnabled(True)
        self.label_74 = QLabel(self.fit_settings_3)
        self.label_74.setObjectName(u"label_74")
        self.label_74.setGeometry(QRect(20, 10, 121, 31))
        font1 = QFont()
        font1.setPointSize(13)
        font1.setBold(True)
        self.label_74.setFont(font1)
        self.layoutWidget_2 = QWidget(self.fit_settings_3)
        self.layoutWidget_2.setObjectName(u"layoutWidget_2")
        self.layoutWidget_2.setGeometry(QRect(20, 50, 381, 224))
        self.verticalLayout_48 = QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_48.setObjectName(u"verticalLayout_48")
        self.verticalLayout_48.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_83 = QHBoxLayout()
        self.horizontalLayout_83.setObjectName(u"horizontalLayout_83")
        self.label_51 = QLabel(self.layoutWidget_2)
        self.label_51.setObjectName(u"label_51")

        self.horizontalLayout_83.addWidget(self.label_51)

        self.cb_fit_negative_2 = QCheckBox(self.layoutWidget_2)
        self.cb_fit_negative_2.setObjectName(u"cb_fit_negative_2")

        self.horizontalLayout_83.addWidget(self.cb_fit_negative_2)


        self.verticalLayout_48.addLayout(self.horizontalLayout_83)

        self.horizontalLayout_84 = QHBoxLayout()
        self.horizontalLayout_84.setObjectName(u"horizontalLayout_84")
        self.label_75 = QLabel(self.layoutWidget_2)
        self.label_75.setObjectName(u"label_75")

        self.horizontalLayout_84.addWidget(self.label_75)

        self.max_iteration_2 = QSpinBox(self.layoutWidget_2)
        self.max_iteration_2.setObjectName(u"max_iteration_2")
        self.max_iteration_2.setMaximum(10000)
        self.max_iteration_2.setValue(200)

        self.horizontalLayout_84.addWidget(self.max_iteration_2)


        self.verticalLayout_48.addLayout(self.horizontalLayout_84)

        self.horizontalLayout_85 = QHBoxLayout()
        self.horizontalLayout_85.setObjectName(u"horizontalLayout_85")
        self.label_76 = QLabel(self.layoutWidget_2)
        self.label_76.setObjectName(u"label_76")

        self.horizontalLayout_85.addWidget(self.label_76)

        self.cbb_fit_methods_2 = QComboBox(self.layoutWidget_2)
        self.cbb_fit_methods_2.setObjectName(u"cbb_fit_methods_2")

        self.horizontalLayout_85.addWidget(self.cbb_fit_methods_2)


        self.verticalLayout_48.addLayout(self.horizontalLayout_85)

        self.horizontalLayout_87 = QHBoxLayout()
        self.horizontalLayout_87.setObjectName(u"horizontalLayout_87")
        self.label_78 = QLabel(self.layoutWidget_2)
        self.label_78.setObjectName(u"label_78")

        self.horizontalLayout_87.addWidget(self.label_78)

        self.xtol_2 = QLineEdit(self.layoutWidget_2)
        self.xtol_2.setObjectName(u"xtol_2")
        self.xtol_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_87.addWidget(self.xtol_2)


        self.verticalLayout_48.addLayout(self.horizontalLayout_87)

        self.verticalSpacer_17 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_48.addItem(self.verticalSpacer_17)

        self.btn_open_fitspy_3 = QPushButton(self.layoutWidget_2)
        self.btn_open_fitspy_3.setObjectName(u"btn_open_fitspy_3")
        self.btn_open_fitspy_3.setMinimumSize(QSize(0, 30))
        self.btn_open_fitspy_3.setMaximumSize(QSize(16777215, 30))

        self.verticalLayout_48.addWidget(self.btn_open_fitspy_3)

        self.l_defaut_folder_model_3 = QLineEdit(self.fit_settings_3)
        self.l_defaut_folder_model_3.setObjectName(u"l_defaut_folder_model_3")
        self.l_defaut_folder_model_3.setGeometry(QRect(160, 320, 481, 21))
        self.btn_default_folder_model_3 = QPushButton(self.fit_settings_3)
        self.btn_default_folder_model_3.setObjectName(u"btn_default_folder_model_3")
        self.btn_default_folder_model_3.setGeometry(QRect(20, 320, 121, 21))
        self.btn_refresh_model_folder_3 = QPushButton(self.fit_settings_3)
        self.btn_refresh_model_folder_3.setObjectName(u"btn_refresh_model_folder_3")
        self.btn_refresh_model_folder_3.setGeometry(QRect(650, 320, 75, 23))
        self.tabWidget_3.addTab(self.fit_settings_3, "")

        self.verticalLayout_66.addWidget(self.tabWidget_3)

        self.splitter_3.addWidget(self.bottom_widget_4)

        self.horizontalLayout_131.addWidget(self.splitter_3)

        self.sidebar_3 = QFrame(self.tab_spectra)
        self.sidebar_3.setObjectName(u"sidebar_3")
        sizePolicy1.setHeightForWidth(self.sidebar_3.sizePolicy().hasHeightForWidth())
        self.sidebar_3.setSizePolicy(sizePolicy1)
        self.sidebar_3.setFrameShape(QFrame.StyledPanel)
        self.sidebar_3.setFrameShadow(QFrame.Raised)
        self.sidebar_3.setLineWidth(0)
        self.verticalLayout_59 = QVBoxLayout(self.sidebar_3)
        self.verticalLayout_59.setObjectName(u"verticalLayout_59")
        self.verticalLayout_59.setContentsMargins(3, 3, 3, 3)
        self.groupBox_12 = QGroupBox(self.sidebar_3)
        self.groupBox_12.setObjectName(u"groupBox_12")
        sizePolicy1.setHeightForWidth(self.groupBox_12.sizePolicy().hasHeightForWidth())
        self.groupBox_12.setSizePolicy(sizePolicy1)
        self.groupBox_12.setMinimumSize(QSize(290, 0))
        self.verticalLayout_61 = QVBoxLayout(self.groupBox_12)
        self.verticalLayout_61.setObjectName(u"verticalLayout_61")
        self.verticalLayout_61.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_103 = QHBoxLayout()
        self.horizontalLayout_103.setObjectName(u"horizontalLayout_103")
        self.btn_sel_all_3 = QPushButton(self.groupBox_12)
        self.btn_sel_all_3.setObjectName(u"btn_sel_all_3")
        sizePolicy3.setHeightForWidth(self.btn_sel_all_3.sizePolicy().hasHeightForWidth())
        self.btn_sel_all_3.setSizePolicy(sizePolicy3)
        self.btn_sel_all_3.setMaximumSize(QSize(30, 16777215))
        icon19 = QIcon()
        icon19.addFile(u":/icon/iconpack/select-all.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_sel_all_3.setIcon(icon19)
        self.btn_sel_all_3.setIconSize(QSize(22, 22))

        self.horizontalLayout_103.addWidget(self.btn_sel_all_3)

        self.btn_remove_spectrum = QPushButton(self.groupBox_12)
        self.btn_remove_spectrum.setObjectName(u"btn_remove_spectrum")
        sizePolicy5.setHeightForWidth(self.btn_remove_spectrum.sizePolicy().hasHeightForWidth())
        self.btn_remove_spectrum.setSizePolicy(sizePolicy5)
        self.btn_remove_spectrum.setMinimumSize(QSize(30, 0))
        self.btn_remove_spectrum.setMaximumSize(QSize(30, 16777215))
        icon20 = QIcon()
        icon20.addFile(u":/icon/iconpack/icons8-trash-128.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_remove_spectrum.setIcon(icon20)
        self.btn_remove_spectrum.setIconSize(QSize(22, 22))

        self.horizontalLayout_103.addWidget(self.btn_remove_spectrum)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_103.addItem(self.horizontalSpacer_9)

        self.btn_init_3 = QPushButton(self.groupBox_12)
        self.btn_init_3.setObjectName(u"btn_init_3")
        sizePolicy3.setHeightForWidth(self.btn_init_3.sizePolicy().hasHeightForWidth())
        self.btn_init_3.setSizePolicy(sizePolicy3)
        self.btn_init_3.setMinimumSize(QSize(70, 0))
        self.btn_init_3.setMaximumSize(QSize(70, 16777215))

        self.horizontalLayout_103.addWidget(self.btn_init_3)

        self.btn_show_stats_3 = QPushButton(self.groupBox_12)
        self.btn_show_stats_3.setObjectName(u"btn_show_stats_3")
        sizePolicy3.setHeightForWidth(self.btn_show_stats_3.sizePolicy().hasHeightForWidth())
        self.btn_show_stats_3.setSizePolicy(sizePolicy3)
        self.btn_show_stats_3.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_103.addWidget(self.btn_show_stats_3)


        self.verticalLayout_61.addLayout(self.horizontalLayout_103)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.checkBox = QCheckBox(self.groupBox_12)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setChecked(True)

        self.horizontalLayout_16.addWidget(self.checkBox)


        self.verticalLayout_61.addLayout(self.horizontalLayout_16)

        self.listbox_layout = QVBoxLayout()
        self.listbox_layout.setObjectName(u"listbox_layout")

        self.verticalLayout_61.addLayout(self.listbox_layout)

        self.horizontalLayout_104 = QHBoxLayout()
        self.horizontalLayout_104.setObjectName(u"horizontalLayout_104")
        self.item_count_label_3 = QLabel(self.groupBox_12)
        self.item_count_label_3.setObjectName(u"item_count_label_3")

        self.horizontalLayout_104.addWidget(self.item_count_label_3)

        self.horizontalSpacer_23 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_104.addItem(self.horizontalSpacer_23)


        self.verticalLayout_61.addLayout(self.horizontalLayout_104)


        self.verticalLayout_59.addWidget(self.groupBox_12)


        self.horizontalLayout_131.addWidget(self.sidebar_3)

        self.tabWidget.addTab(self.tab_spectra, "")
        self.tab_maps = QWidget()
        self.tab_maps.setObjectName(u"tab_maps")
        self.gridLayout_5 = QGridLayout(self.tab_maps)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(5, 5, 5, 5)
        self.splitter = QSplitter(self.tab_maps)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.splitter.setHandleWidth(10)
        self.upper_frame = QFrame(self.splitter)
        self.upper_frame.setObjectName(u"upper_frame")
        self.horizontalLayout_27 = QHBoxLayout(self.upper_frame)
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.horizontalLayout_27.setContentsMargins(3, 0, 3, 3)
        self.Upper_zone = QHBoxLayout()
        self.Upper_zone.setSpacing(0)
        self.Upper_zone.setObjectName(u"Upper_zone")
        self.verticalLayout_26 = QVBoxLayout()
        self.verticalLayout_26.setObjectName(u"verticalLayout_26")
        self.verticalLayout_26.setContentsMargins(0, -1, 10, -1)
        self.spectre_view_frame_ = QFrame(self.upper_frame)
        self.spectre_view_frame_.setObjectName(u"spectre_view_frame_")
        self.spectre_view_frame_.setFrameShape(QFrame.StyledPanel)
        self.spectre_view_frame_.setFrameShadow(QFrame.Raised)
        self.verticalLayout_16 = QVBoxLayout(self.spectre_view_frame_)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.verticalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.QVBoxlayout = QVBoxLayout()
        self.QVBoxlayout.setSpacing(6)
        self.QVBoxlayout.setObjectName(u"QVBoxlayout")

        self.verticalLayout_16.addLayout(self.QVBoxlayout)


        self.verticalLayout_26.addWidget(self.spectre_view_frame_)

        self.bottom_frame = QHBoxLayout()
        self.bottom_frame.setSpacing(10)
        self.bottom_frame.setObjectName(u"bottom_frame")
        self.bottom_frame.setSizeConstraint(QLayout.SetMaximumSize)
        self.bottom_frame.setContentsMargins(2, 2, 2, 2)
        self.toolbar_frame = QHBoxLayout()
        self.toolbar_frame.setObjectName(u"toolbar_frame")

        self.bottom_frame.addLayout(self.toolbar_frame)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottom_frame.addItem(self.horizontalSpacer_7)

        self.rdbtn_baseline = QRadioButton(self.upper_frame)
        self.rdbtn_baseline.setObjectName(u"rdbtn_baseline")
        self.rdbtn_baseline.setChecked(True)

        self.bottom_frame.addWidget(self.rdbtn_baseline)

        self.rdbtn_peak = QRadioButton(self.upper_frame)
        self.rdbtn_peak.setObjectName(u"rdbtn_peak")
        self.rdbtn_peak.setChecked(False)

        self.bottom_frame.addWidget(self.rdbtn_peak)

        self.horizontalSpacer_15 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottom_frame.addItem(self.horizontalSpacer_15)

        self.rsquared_1 = QLabel(self.upper_frame)
        self.rsquared_1.setObjectName(u"rsquared_1")
        self.rsquared_1.setMinimumSize(QSize(80, 0))
        self.rsquared_1.setMaximumSize(QSize(80, 16777215))

        self.bottom_frame.addWidget(self.rsquared_1)

        self.btn_copy_fig = QPushButton(self.upper_frame)
        self.btn_copy_fig.setObjectName(u"btn_copy_fig")
        self.btn_copy_fig.setIcon(icon11)
        self.btn_copy_fig.setIconSize(QSize(24, 24))

        self.bottom_frame.addWidget(self.btn_copy_fig)

        self.label_63 = QLabel(self.upper_frame)
        self.label_63.setObjectName(u"label_63")
        sizePolicy4.setHeightForWidth(self.label_63.sizePolicy().hasHeightForWidth())
        self.label_63.setSizePolicy(sizePolicy4)
        self.label_63.setMinimumSize(QSize(20, 0))
        self.label_63.setMaximumSize(QSize(20, 16777215))

        self.bottom_frame.addWidget(self.label_63)

        self.sb_dpi_spectra = QSpinBox(self.upper_frame)
        self.sb_dpi_spectra.setObjectName(u"sb_dpi_spectra")
        self.sb_dpi_spectra.setMinimum(100)
        self.sb_dpi_spectra.setMaximum(200)
        self.sb_dpi_spectra.setSingleStep(10)

        self.bottom_frame.addWidget(self.sb_dpi_spectra)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.bottom_frame.addItem(self.horizontalSpacer_8)

        self.bottom_frame.setStretch(0, 50)
        self.bottom_frame.setStretch(1, 25)
        self.bottom_frame.setStretch(9, 2)

        self.verticalLayout_26.addLayout(self.bottom_frame)

        self.verticalLayout_26.setStretch(0, 75)
        self.verticalLayout_26.setStretch(1, 25)

        self.Upper_zone.addLayout(self.verticalLayout_26)

        self.widget_7 = QWidget(self.upper_frame)
        self.widget_7.setObjectName(u"widget_7")
        sizePolicy1.setHeightForWidth(self.widget_7.sizePolicy().hasHeightForWidth())
        self.widget_7.setSizePolicy(sizePolicy1)
        self.widget_7.setMinimumSize(QSize(300, 0))
        self.widget_7.setMaximumSize(QSize(320, 16777215))
        self.verticalLayout_13 = QVBoxLayout(self.widget_7)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(2, 0, 2, 0)
        self.horizontalLayout_69 = QHBoxLayout()
        self.horizontalLayout_69.setObjectName(u"horizontalLayout_69")
        self.horizontalSpacer_21 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_69.addItem(self.horizontalSpacer_21)

        self.measurement_sites = QFrame(self.widget_7)
        self.measurement_sites.setObjectName(u"measurement_sites")
        sizePolicy5.setHeightForWidth(self.measurement_sites.sizePolicy().hasHeightForWidth())
        self.measurement_sites.setSizePolicy(sizePolicy5)
        self.measurement_sites.setMinimumSize(QSize(320, 330))
        self.measurement_sites.setMaximumSize(QSize(320, 330))
        self.measurement_sites.setFrameShape(QFrame.StyledPanel)
        self.measurement_sites.setFrameShadow(QFrame.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.measurement_sites)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(5, 5, 5, 5)

        self.horizontalLayout_69.addWidget(self.measurement_sites)


        self.verticalLayout_13.addLayout(self.horizontalLayout_69)

        self.horizontalLayout_50 = QHBoxLayout()
        self.horizontalLayout_50.setObjectName(u"horizontalLayout_50")
        self.horizontalSpacer_18 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_50.addItem(self.horizontalSpacer_18)

        self.rdbt_show_2Dmap = QRadioButton(self.widget_7)
        self.rdbt_show_2Dmap.setObjectName(u"rdbt_show_2Dmap")
        self.rdbt_show_2Dmap.setChecked(False)

        self.horizontalLayout_50.addWidget(self.rdbt_show_2Dmap)

        self.rdbt_show_wafer = QRadioButton(self.widget_7)
        self.rdbt_show_wafer.setObjectName(u"rdbt_show_wafer")
        self.rdbt_show_wafer.setChecked(True)

        self.horizontalLayout_50.addWidget(self.rdbt_show_wafer)

        self.cbb_wafer_size = QComboBox(self.widget_7)
        self.cbb_wafer_size.setObjectName(u"cbb_wafer_size")

        self.horizontalLayout_50.addWidget(self.cbb_wafer_size)


        self.verticalLayout_13.addLayout(self.horizontalLayout_50)

        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_13.addItem(self.verticalSpacer_5)

        self.view_options_box = QGroupBox(self.widget_7)
        self.view_options_box.setObjectName(u"view_options_box")
        self.view_options_box.setMaximumSize(QSize(320, 16777215))
        self.gridLayout_6 = QGridLayout(self.view_options_box)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.cb_residual = QCheckBox(self.view_options_box)
        self.cb_residual.setObjectName(u"cb_residual")
        self.cb_residual.setChecked(False)

        self.gridLayout_6.addWidget(self.cb_residual, 1, 1, 1, 1)

        self.cb_filled = QCheckBox(self.view_options_box)
        self.cb_filled.setObjectName(u"cb_filled")
        self.cb_filled.setChecked(True)

        self.gridLayout_6.addWidget(self.cb_filled, 0, 2, 1, 1)

        self.cb_bestfit = QCheckBox(self.view_options_box)
        self.cb_bestfit.setObjectName(u"cb_bestfit")
        self.cb_bestfit.setChecked(True)

        self.gridLayout_6.addWidget(self.cb_bestfit, 0, 1, 1, 1)

        self.cb_legend = QCheckBox(self.view_options_box)
        self.cb_legend.setObjectName(u"cb_legend")
        self.cb_legend.setEnabled(True)
        self.cb_legend.setChecked(False)

        self.gridLayout_6.addWidget(self.cb_legend, 0, 0, 1, 1)

        self.cb_raw = QCheckBox(self.view_options_box)
        self.cb_raw.setObjectName(u"cb_raw")
        self.cb_raw.setChecked(False)

        self.gridLayout_6.addWidget(self.cb_raw, 1, 0, 1, 1)

        self.cb_colors = QCheckBox(self.view_options_box)
        self.cb_colors.setObjectName(u"cb_colors")
        self.cb_colors.setChecked(True)

        self.gridLayout_6.addWidget(self.cb_colors, 1, 2, 1, 1)

        self.cb_peaks = QCheckBox(self.view_options_box)
        self.cb_peaks.setObjectName(u"cb_peaks")
        self.cb_peaks.setChecked(False)

        self.gridLayout_6.addWidget(self.cb_peaks, 0, 3, 1, 1)

        self.cb_normalize = QCheckBox(self.view_options_box)
        self.cb_normalize.setObjectName(u"cb_normalize")

        self.gridLayout_6.addWidget(self.cb_normalize, 1, 3, 1, 1)


        self.verticalLayout_13.addWidget(self.view_options_box)


        self.Upper_zone.addWidget(self.widget_7)

        self.Upper_zone.setStretch(0, 75)

        self.horizontalLayout_27.addLayout(self.Upper_zone)

        self.splitter.addWidget(self.upper_frame)
        self.bottom_widget_2 = QWidget(self.splitter)
        self.bottom_widget_2.setObjectName(u"bottom_widget_2")
        self.verticalLayout_25 = QVBoxLayout(self.bottom_widget_2)
        self.verticalLayout_25.setObjectName(u"verticalLayout_25")
        self.verticalLayout_25.setContentsMargins(3, 3, 3, 0)
        self.tabWidget_2 = QTabWidget(self.bottom_widget_2)
        self.tabWidget_2.setObjectName(u"tabWidget_2")
        self.tabWidget_2.setEnabled(True)
        self.fit_model_editor = QWidget()
        self.fit_model_editor.setObjectName(u"fit_model_editor")
        self.fit_model_editor.setEnabled(True)
        self.verticalLayout_14 = QVBoxLayout(self.fit_model_editor)
        self.verticalLayout_14.setSpacing(6)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setSpacing(5)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(-1, 5, 5, 5)
        self.widget_17 = QWidget(self.fit_model_editor)
        self.widget_17.setObjectName(u"widget_17")
        self.horizontalLayout_44 = QHBoxLayout(self.widget_17)
        self.horizontalLayout_44.setSpacing(6)
        self.horizontalLayout_44.setObjectName(u"horizontalLayout_44")
        self.horizontalLayout_44.setContentsMargins(2, 2, 2, 2)
        self.scrollArea_3 = QScrollArea(self.widget_17)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setMinimumSize(QSize(430, 100))
        self.scrollArea_3.setMaximumSize(QSize(430, 16777215))
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 401, 379))
        self.verticalLayout_38 = QVBoxLayout(self.scrollAreaWidgetContents_3)
        self.verticalLayout_38.setSpacing(10)
        self.verticalLayout_38.setObjectName(u"verticalLayout_38")
        self.verticalLayout_38.setContentsMargins(10, 10, 10, 10)
        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.btn_cosmis_ray = QPushButton(self.scrollAreaWidgetContents_3)
        self.btn_cosmis_ray.setObjectName(u"btn_cosmis_ray")
        self.btn_cosmis_ray.setMinimumSize(QSize(80, 0))
        self.btn_cosmis_ray.setMaximumSize(QSize(150, 16777215))

        self.horizontalLayout_18.addWidget(self.btn_cosmis_ray)

        self.horizontalSpacer_56 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_18.addItem(self.horizontalSpacer_56)

        self.label_99 = QLabel(self.scrollAreaWidgetContents_3)
        self.label_99.setObjectName(u"label_99")

        self.horizontalLayout_18.addWidget(self.label_99)

        self.cbb_xaxis_unit2 = QComboBox(self.scrollAreaWidgetContents_3)
        self.cbb_xaxis_unit2.setObjectName(u"cbb_xaxis_unit2")

        self.horizontalLayout_18.addWidget(self.cbb_xaxis_unit2)


        self.verticalLayout_38.addLayout(self.horizontalLayout_18)

        self.groupBox_4 = QGroupBox(self.scrollAreaWidgetContents_3)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.verticalLayout_36 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_36.setSpacing(5)
        self.verticalLayout_36.setObjectName(u"verticalLayout_36")
        self.verticalLayout_36.setContentsMargins(2, 2, 2, 2)
        self.label_54 = QLabel(self.groupBox_4)
        self.label_54.setObjectName(u"label_54")
        self.label_54.setFont(font)

        self.verticalLayout_36.addWidget(self.label_54)

        self.horizontalLayout_58 = QHBoxLayout()
        self.horizontalLayout_58.setSpacing(5)
        self.horizontalLayout_58.setObjectName(u"horizontalLayout_58")
        self.horizontalLayout_58.setContentsMargins(2, 2, 2, 2)
        self.label_61 = QLabel(self.groupBox_4)
        self.label_61.setObjectName(u"label_61")

        self.horizontalLayout_58.addWidget(self.label_61)

        self.horizontalSpacer_30 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_58.addItem(self.horizontalSpacer_30)

        self.range_min = QLineEdit(self.groupBox_4)
        self.range_min.setObjectName(u"range_min")

        self.horizontalLayout_58.addWidget(self.range_min)

        self.label_62 = QLabel(self.groupBox_4)
        self.label_62.setObjectName(u"label_62")

        self.horizontalLayout_58.addWidget(self.label_62)

        self.range_max = QLineEdit(self.groupBox_4)
        self.range_max.setObjectName(u"range_max")

        self.horizontalLayout_58.addWidget(self.range_max)

        self.range_apply = QPushButton(self.groupBox_4)
        self.range_apply.setObjectName(u"range_apply")

        self.horizontalLayout_58.addWidget(self.range_apply)


        self.verticalLayout_36.addLayout(self.horizontalLayout_58)


        self.verticalLayout_38.addWidget(self.groupBox_4)

        self.label_59 = QLabel(self.scrollAreaWidgetContents_3)
        self.label_59.setObjectName(u"label_59")

        self.verticalLayout_38.addWidget(self.label_59)

        self.baseline = QGroupBox(self.scrollAreaWidgetContents_3)
        self.baseline.setObjectName(u"baseline")
        self.verticalLayout_6 = QVBoxLayout(self.baseline)
        self.verticalLayout_6.setSpacing(5)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(2, 2, 2, 2)
        self.label_52 = QLabel(self.baseline)
        self.label_52.setObjectName(u"label_52")
        self.label_52.setFont(font)

        self.verticalLayout_6.addWidget(self.label_52)

        self.horizontalLayout_37 = QHBoxLayout()
        self.horizontalLayout_37.setSpacing(5)
        self.horizontalLayout_37.setObjectName(u"horizontalLayout_37")
        self.horizontalLayout_37.setContentsMargins(2, 2, 2, 2)
        self.rbtn_linear = QRadioButton(self.baseline)
        self.rbtn_linear.setObjectName(u"rbtn_linear")
        sizePolicy3.setHeightForWidth(self.rbtn_linear.sizePolicy().hasHeightForWidth())
        self.rbtn_linear.setSizePolicy(sizePolicy3)
        self.rbtn_linear.setChecked(True)

        self.horizontalLayout_37.addWidget(self.rbtn_linear)

        self.rbtn_polynomial = QRadioButton(self.baseline)
        self.rbtn_polynomial.setObjectName(u"rbtn_polynomial")
        sizePolicy3.setHeightForWidth(self.rbtn_polynomial.sizePolicy().hasHeightForWidth())
        self.rbtn_polynomial.setSizePolicy(sizePolicy3)

        self.horizontalLayout_37.addWidget(self.rbtn_polynomial)

        self.degre = QSpinBox(self.baseline)
        self.degre.setObjectName(u"degre")
        self.degre.setMinimum(1)

        self.horizontalLayout_37.addWidget(self.degre)

        self.cb_attached = QCheckBox(self.baseline)
        self.cb_attached.setObjectName(u"cb_attached")
        sizePolicy3.setHeightForWidth(self.cb_attached.sizePolicy().hasHeightForWidth())
        self.cb_attached.setSizePolicy(sizePolicy3)
        self.cb_attached.setChecked(True)

        self.horizontalLayout_37.addWidget(self.cb_attached)

        self.horizontalSpacer_32 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_37.addItem(self.horizontalSpacer_32)

        self.label_37 = QLabel(self.baseline)
        self.label_37.setObjectName(u"label_37")

        self.horizontalLayout_37.addWidget(self.label_37)

        self.noise = QDoubleSpinBox(self.baseline)
        self.noise.setObjectName(u"noise")
        self.noise.setDecimals(0)
        self.noise.setValue(5.000000000000000)

        self.horizontalLayout_37.addWidget(self.noise)

        self.horizontalLayout_37.setStretch(0, 25)
        self.horizontalLayout_37.setStretch(1, 25)

        self.verticalLayout_6.addLayout(self.horizontalLayout_37)

        self.horizontalLayout_57 = QHBoxLayout()
        self.horizontalLayout_57.setSpacing(5)
        self.horizontalLayout_57.setObjectName(u"horizontalLayout_57")
        self.horizontalLayout_57.setContentsMargins(2, 2, 2, 2)
        self.btn_undo_baseline = QPushButton(self.baseline)
        self.btn_undo_baseline.setObjectName(u"btn_undo_baseline")
        self.btn_undo_baseline.setIcon(icon12)

        self.horizontalLayout_57.addWidget(self.btn_undo_baseline)

        self.horizontalSpacer_22 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_57.addItem(self.horizontalSpacer_22)

        self.btn_copy_bl = QPushButton(self.baseline)
        self.btn_copy_bl.setObjectName(u"btn_copy_bl")
        self.btn_copy_bl.setIcon(icon11)

        self.horizontalLayout_57.addWidget(self.btn_copy_bl)

        self.btn_paste_bl = QPushButton(self.baseline)
        self.btn_paste_bl.setObjectName(u"btn_paste_bl")
        self.btn_paste_bl.setIcon(icon13)

        self.horizontalLayout_57.addWidget(self.btn_paste_bl)

        self.sub_baseline = QPushButton(self.baseline)
        self.sub_baseline.setObjectName(u"sub_baseline")

        self.horizontalLayout_57.addWidget(self.sub_baseline)


        self.verticalLayout_6.addLayout(self.horizontalLayout_57)


        self.verticalLayout_38.addWidget(self.baseline)

        self.label_60 = QLabel(self.scrollAreaWidgetContents_3)
        self.label_60.setObjectName(u"label_60")

        self.verticalLayout_38.addWidget(self.label_60)

        self.peaks = QGroupBox(self.scrollAreaWidgetContents_3)
        self.peaks.setObjectName(u"peaks")
        self.verticalLayout_34 = QVBoxLayout(self.peaks)
        self.verticalLayout_34.setSpacing(5)
        self.verticalLayout_34.setObjectName(u"verticalLayout_34")
        self.verticalLayout_34.setContentsMargins(2, 2, 2, 2)
        self.label_57 = QLabel(self.peaks)
        self.label_57.setObjectName(u"label_57")
        self.label_57.setFont(font)

        self.verticalLayout_34.addWidget(self.label_57)

        self.horizontalLayout_56 = QHBoxLayout()
        self.horizontalLayout_56.setSpacing(5)
        self.horizontalLayout_56.setObjectName(u"horizontalLayout_56")
        self.horizontalLayout_56.setContentsMargins(2, 2, 2, 2)
        self.label_41 = QLabel(self.peaks)
        self.label_41.setObjectName(u"label_41")

        self.horizontalLayout_56.addWidget(self.label_41)

        self.horizontalSpacer_31 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_56.addItem(self.horizontalSpacer_31)

        self.cbb_fit_models = QComboBox(self.peaks)
        self.cbb_fit_models.setObjectName(u"cbb_fit_models")

        self.horizontalLayout_56.addWidget(self.cbb_fit_models)

        self.clear_peaks = QPushButton(self.peaks)
        self.clear_peaks.setObjectName(u"clear_peaks")
        self.clear_peaks.setIcon(icon12)

        self.horizontalLayout_56.addWidget(self.clear_peaks)

        self.horizontalLayout_56.setStretch(2, 65)

        self.verticalLayout_34.addLayout(self.horizontalLayout_56)


        self.verticalLayout_38.addWidget(self.peaks)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_38.addItem(self.verticalSpacer)

        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)

        self.horizontalLayout_44.addWidget(self.scrollArea_3)

        self.verticalLayout_33 = QVBoxLayout()
        self.verticalLayout_33.setObjectName(u"verticalLayout_33")
        self.peak_table = QGroupBox(self.widget_17)
        self.peak_table.setObjectName(u"peak_table")
        self.horizontalLayout_26 = QHBoxLayout(self.peak_table)
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.scrollArea_5 = QScrollArea(self.peak_table)
        self.scrollArea_5.setObjectName(u"scrollArea_5")
        self.scrollArea_5.setWidgetResizable(True)
        self.scrollAreaWidgetContents_5 = QWidget()
        self.scrollAreaWidgetContents_5.setObjectName(u"scrollAreaWidgetContents_5")
        self.scrollAreaWidgetContents_5.setGeometry(QRect(0, 0, 98, 40))
        self.verticalLayout_35 = QVBoxLayout(self.scrollAreaWidgetContents_5)
        self.verticalLayout_35.setObjectName(u"verticalLayout_35")
        self.verticalLayout_311 = QVBoxLayout()
        self.verticalLayout_311.setObjectName(u"verticalLayout_311")
        self.horizontalLayout_53 = QHBoxLayout()
        self.horizontalLayout_53.setObjectName(u"horizontalLayout_53")
        self.peak_table1 = QHBoxLayout()
        self.peak_table1.setObjectName(u"peak_table1")

        self.horizontalLayout_53.addLayout(self.peak_table1)

        self.horizontalLayout_54 = QHBoxLayout()
        self.horizontalLayout_54.setObjectName(u"horizontalLayout_54")

        self.horizontalLayout_53.addLayout(self.horizontalLayout_54)


        self.verticalLayout_311.addLayout(self.horizontalLayout_53)

        self.verticalSpacer_6 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_311.addItem(self.verticalSpacer_6)


        self.verticalLayout_35.addLayout(self.verticalLayout_311)

        self.scrollArea_5.setWidget(self.scrollAreaWidgetContents_5)

        self.horizontalLayout_26.addWidget(self.scrollArea_5)


        self.verticalLayout_33.addWidget(self.peak_table)

        self.horizontalLayout_70 = QHBoxLayout()
        self.horizontalLayout_70.setObjectName(u"horizontalLayout_70")
        self.btn_fit = QPushButton(self.widget_17)
        self.btn_fit.setObjectName(u"btn_fit")
        self.btn_fit.setMinimumSize(QSize(50, 50))
        self.btn_fit.setMaximumSize(QSize(50, 50))

        self.horizontalLayout_70.addWidget(self.btn_fit)

        self.verticalLayout_22 = QVBoxLayout()
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.horizontalLayout_51 = QHBoxLayout()
        self.horizontalLayout_51.setSpacing(5)
        self.horizontalLayout_51.setObjectName(u"horizontalLayout_51")
        self.horizontalLayout_51.setContentsMargins(5, 2, 5, 2)
        self.btn_copy_fit_model = QPushButton(self.widget_17)
        self.btn_copy_fit_model.setObjectName(u"btn_copy_fit_model")
        self.btn_copy_fit_model.setIcon(icon11)

        self.horizontalLayout_51.addWidget(self.btn_copy_fit_model)

        self.lbl_copied_fit_model = QLabel(self.widget_17)
        self.lbl_copied_fit_model.setObjectName(u"lbl_copied_fit_model")
        sizePolicy4.setHeightForWidth(self.lbl_copied_fit_model.sizePolicy().hasHeightForWidth())
        self.lbl_copied_fit_model.setSizePolicy(sizePolicy4)
        self.lbl_copied_fit_model.setMinimumSize(QSize(50, 0))

        self.horizontalLayout_51.addWidget(self.lbl_copied_fit_model)

        self.btn_paste_fit_model = QPushButton(self.widget_17)
        self.btn_paste_fit_model.setObjectName(u"btn_paste_fit_model")
        self.btn_paste_fit_model.setMinimumSize(QSize(0, 0))
        self.btn_paste_fit_model.setMaximumSize(QSize(16777215, 40))
        self.btn_paste_fit_model.setIcon(icon13)

        self.horizontalLayout_51.addWidget(self.btn_paste_fit_model)

        self.horizontalSpacer_50 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_51.addItem(self.horizontalSpacer_50)

        self.save_model = QPushButton(self.widget_17)
        self.save_model.setObjectName(u"save_model")
        self.save_model.setIcon(icon14)

        self.horizontalLayout_51.addWidget(self.save_model)

        self.horizontalSpacer_17 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_51.addItem(self.horizontalSpacer_17)

        self.cb_limits = QCheckBox(self.widget_17)
        self.cb_limits.setObjectName(u"cb_limits")

        self.horizontalLayout_51.addWidget(self.cb_limits)

        self.cb_expr = QCheckBox(self.widget_17)
        self.cb_expr.setObjectName(u"cb_expr")

        self.horizontalLayout_51.addWidget(self.cb_expr)


        self.verticalLayout_22.addLayout(self.horizontalLayout_51)

        self.horizontalLayout_52 = QHBoxLayout()
        self.horizontalLayout_52.setSpacing(5)
        self.horizontalLayout_52.setObjectName(u"horizontalLayout_52")
        self.horizontalLayout_52.setContentsMargins(5, 2, 5, 2)
        self.label_80 = QLabel(self.widget_17)
        self.label_80.setObjectName(u"label_80")

        self.horizontalLayout_52.addWidget(self.label_80)

        self.cbb_fit_model_list = QComboBox(self.widget_17)
        self.cbb_fit_model_list.setObjectName(u"cbb_fit_model_list")
        self.cbb_fit_model_list.setMinimumSize(QSize(400, 0))
        self.cbb_fit_model_list.setMaximumSize(QSize(400, 16777215))

        self.horizontalLayout_52.addWidget(self.cbb_fit_model_list)

        self.btn_apply_model = QPushButton(self.widget_17)
        self.btn_apply_model.setObjectName(u"btn_apply_model")
        sizePolicy3.setHeightForWidth(self.btn_apply_model.sizePolicy().hasHeightForWidth())
        self.btn_apply_model.setSizePolicy(sizePolicy3)
        self.btn_apply_model.setMinimumSize(QSize(0, 0))
        self.btn_apply_model.setMaximumSize(QSize(85, 32))

        self.horizontalLayout_52.addWidget(self.btn_apply_model)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_52.addItem(self.horizontalSpacer_6)

        self.btn_load_model = QPushButton(self.widget_17)
        self.btn_load_model.setObjectName(u"btn_load_model")
        self.btn_load_model.setMaximumSize(QSize(85, 16777215))

        self.horizontalLayout_52.addWidget(self.btn_load_model)


        self.verticalLayout_22.addLayout(self.horizontalLayout_52)


        self.horizontalLayout_70.addLayout(self.verticalLayout_22)


        self.verticalLayout_33.addLayout(self.horizontalLayout_70)

        self.verticalLayout_33.setStretch(0, 85)
        self.verticalLayout_33.setStretch(1, 15)

        self.horizontalLayout_44.addLayout(self.verticalLayout_33)

        self.horizontalLayout_44.setStretch(1, 60)

        self.horizontalLayout_9.addWidget(self.widget_17)


        self.verticalLayout_14.addLayout(self.horizontalLayout_9)

        self.tabWidget_2.addTab(self.fit_model_editor, "")
        self.collect_fit_data = QWidget()
        self.collect_fit_data.setObjectName(u"collect_fit_data")
        self.verticalLayout_32 = QVBoxLayout(self.collect_fit_data)
        self.verticalLayout_32.setSpacing(6)
        self.verticalLayout_32.setObjectName(u"verticalLayout_32")
        self.verticalLayout_32.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_62 = QHBoxLayout()
        self.horizontalLayout_62.setSpacing(5)
        self.horizontalLayout_62.setObjectName(u"horizontalLayout_62")
        self.horizontalLayout_62.setContentsMargins(5, 5, 5, 5)
        self.scrollArea_9 = QScrollArea(self.collect_fit_data)
        self.scrollArea_9.setObjectName(u"scrollArea_9")
        sizePolicy2.setHeightForWidth(self.scrollArea_9.sizePolicy().hasHeightForWidth())
        self.scrollArea_9.setSizePolicy(sizePolicy2)
        self.scrollArea_9.setMinimumSize(QSize(430, 100))
        self.scrollArea_9.setMaximumSize(QSize(430, 16777215))
        self.scrollArea_9.setWidgetResizable(True)
        self.scrollAreaWidgetContents_9 = QWidget()
        self.scrollAreaWidgetContents_9.setObjectName(u"scrollAreaWidgetContents_9")
        self.scrollAreaWidgetContents_9.setGeometry(QRect(0, 0, 283, 217))
        self.verticalLayout_80 = QVBoxLayout(self.scrollAreaWidgetContents_9)
        self.verticalLayout_80.setSpacing(10)
        self.verticalLayout_80.setObjectName(u"verticalLayout_80")
        self.verticalLayout_80.setContentsMargins(10, 10, 10, 10)
        self.btn_collect_results = QPushButton(self.scrollAreaWidgetContents_9)
        self.btn_collect_results.setObjectName(u"btn_collect_results")
        sizePolicy5.setHeightForWidth(self.btn_collect_results.sizePolicy().hasHeightForWidth())
        self.btn_collect_results.setSizePolicy(sizePolicy5)
        self.btn_collect_results.setMinimumSize(QSize(140, 40))
        self.btn_collect_results.setMaximumSize(QSize(140, 40))
        self.btn_collect_results.setFont(font)
        self.btn_collect_results.setIcon(icon15)
        self.btn_collect_results.setIconSize(QSize(16, 22))

        self.verticalLayout_80.addWidget(self.btn_collect_results)

        self.label_56 = QLabel(self.scrollAreaWidgetContents_9)
        self.label_56.setObjectName(u"label_56")

        self.verticalLayout_80.addWidget(self.label_56)

        self.horizontalLayout_49 = QHBoxLayout()
        self.horizontalLayout_49.setObjectName(u"horizontalLayout_49")
        self.btn_split_fname_2 = QPushButton(self.scrollAreaWidgetContents_9)
        self.btn_split_fname_2.setObjectName(u"btn_split_fname_2")
        sizePolicy5.setHeightForWidth(self.btn_split_fname_2.sizePolicy().hasHeightForWidth())
        self.btn_split_fname_2.setSizePolicy(sizePolicy5)
        self.btn_split_fname_2.setMinimumSize(QSize(40, 0))
        self.btn_split_fname_2.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_49.addWidget(self.btn_split_fname_2)

        self.cbb_split_fname_2 = QComboBox(self.scrollAreaWidgetContents_9)
        self.cbb_split_fname_2.setObjectName(u"cbb_split_fname_2")
        sizePolicy5.setHeightForWidth(self.cbb_split_fname_2.sizePolicy().hasHeightForWidth())
        self.cbb_split_fname_2.setSizePolicy(sizePolicy5)
        self.cbb_split_fname_2.setMinimumSize(QSize(120, 0))
        self.cbb_split_fname_2.setMaximumSize(QSize(120, 16777215))

        self.horizontalLayout_49.addWidget(self.cbb_split_fname_2)

        self.ent_col_name_2 = QLineEdit(self.scrollAreaWidgetContents_9)
        self.ent_col_name_2.setObjectName(u"ent_col_name_2")

        self.horizontalLayout_49.addWidget(self.ent_col_name_2)

        self.btn_add_col_2 = QPushButton(self.scrollAreaWidgetContents_9)
        self.btn_add_col_2.setObjectName(u"btn_add_col_2")
        self.btn_add_col_2.setMinimumSize(QSize(60, 0))
        self.btn_add_col_2.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout_49.addWidget(self.btn_add_col_2)

        self.horizontalLayout_49.setStretch(2, 40)
        self.horizontalLayout_49.setStretch(3, 20)

        self.verticalLayout_80.addLayout(self.horizontalLayout_49)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.ent_send_df_to_viz2 = QLineEdit(self.scrollAreaWidgetContents_9)
        self.ent_send_df_to_viz2.setObjectName(u"ent_send_df_to_viz2")

        self.horizontalLayout_7.addWidget(self.ent_send_df_to_viz2)

        self.btn_send_to_viz2 = QPushButton(self.scrollAreaWidgetContents_9)
        self.btn_send_to_viz2.setObjectName(u"btn_send_to_viz2")

        self.horizontalLayout_7.addWidget(self.btn_send_to_viz2)


        self.verticalLayout_80.addLayout(self.horizontalLayout_7)

        self.verticalSpacer_13 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_80.addItem(self.verticalSpacer_13)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_20.addItem(self.horizontalSpacer_3)

        self.btn_view_df_2 = QPushButton(self.scrollAreaWidgetContents_9)
        self.btn_view_df_2.setObjectName(u"btn_view_df_2")
        sizePolicy5.setHeightForWidth(self.btn_view_df_2.sizePolicy().hasHeightForWidth())
        self.btn_view_df_2.setSizePolicy(sizePolicy5)
        self.btn_view_df_2.setMinimumSize(QSize(30, 0))
        self.btn_view_df_2.setMaximumSize(QSize(30, 16777215))
        self.btn_view_df_2.setIcon(icon16)
        self.btn_view_df_2.setIconSize(QSize(22, 22))

        self.horizontalLayout_20.addWidget(self.btn_view_df_2)

        self.btn_save_fit_results = QPushButton(self.scrollAreaWidgetContents_9)
        self.btn_save_fit_results.setObjectName(u"btn_save_fit_results")
        sizePolicy5.setHeightForWidth(self.btn_save_fit_results.sizePolicy().hasHeightForWidth())
        self.btn_save_fit_results.setSizePolicy(sizePolicy5)
        self.btn_save_fit_results.setMinimumSize(QSize(30, 0))
        self.btn_save_fit_results.setMaximumSize(QSize(30, 16777215))
        self.btn_save_fit_results.setIcon(icon17)
        self.btn_save_fit_results.setIconSize(QSize(22, 22))

        self.horizontalLayout_20.addWidget(self.btn_save_fit_results)

        self.btn_open_fit_results = QPushButton(self.scrollAreaWidgetContents_9)
        self.btn_open_fit_results.setObjectName(u"btn_open_fit_results")
        sizePolicy3.setHeightForWidth(self.btn_open_fit_results.sizePolicy().hasHeightForWidth())
        self.btn_open_fit_results.setSizePolicy(sizePolicy3)
        self.btn_open_fit_results.setMaximumSize(QSize(30, 16777215))
        self.btn_open_fit_results.setIcon(icon18)
        self.btn_open_fit_results.setIconSize(QSize(22, 22))

        self.horizontalLayout_20.addWidget(self.btn_open_fit_results)


        self.verticalLayout_80.addLayout(self.horizontalLayout_20)

        self.scrollArea_9.setWidget(self.scrollAreaWidgetContents_9)

        self.horizontalLayout_62.addWidget(self.scrollArea_9)

        self.verticalLayout_37 = QVBoxLayout()
        self.verticalLayout_37.setObjectName(u"verticalLayout_37")
        self.verticalLayout_37.setContentsMargins(15, -1, -1, -1)
        self.layout_df_table = QVBoxLayout()
        self.layout_df_table.setObjectName(u"layout_df_table")

        self.verticalLayout_37.addLayout(self.layout_df_table)

        self.groupBox_3 = QGroupBox(self.collect_fit_data)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.horizontalLayout_36 = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_36.setSpacing(9)
        self.horizontalLayout_36.setObjectName(u"horizontalLayout_36")
        self.horizontalLayout_36.setContentsMargins(3, 3, 3, 3)

        self.verticalLayout_37.addWidget(self.groupBox_3)


        self.horizontalLayout_62.addLayout(self.verticalLayout_37)


        self.verticalLayout_32.addLayout(self.horizontalLayout_62)

        self.tabWidget_2.addTab(self.collect_fit_data, "")
        self.fit_settings = QWidget()
        self.fit_settings.setObjectName(u"fit_settings")
        self.fit_settings.setEnabled(True)
        self.groupBox_8 = QGroupBox(self.fit_settings)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.groupBox_8.setGeometry(QRect(780, 30, 376, 110))
        self.verticalLayout_28 = QVBoxLayout(self.groupBox_8)
        self.verticalLayout_28.setObjectName(u"verticalLayout_28")
        self.horizontalLayout_61 = QHBoxLayout()
        self.horizontalLayout_61.setObjectName(u"horizontalLayout_61")
        self.radioButton_3 = QRadioButton(self.groupBox_8)
        self.radioButton_3.setObjectName(u"radioButton_3")
        sizePolicy3.setHeightForWidth(self.radioButton_3.sizePolicy().hasHeightForWidth())
        self.radioButton_3.setSizePolicy(sizePolicy3)
        self.radioButton_3.setChecked(True)

        self.horizontalLayout_61.addWidget(self.radioButton_3)

        self.label_45 = QLabel(self.groupBox_8)
        self.label_45.setObjectName(u"label_45")

        self.horizontalLayout_61.addWidget(self.label_45)

        self.horizontalLayout_60 = QHBoxLayout()
        self.horizontalLayout_60.setObjectName(u"horizontalLayout_60")
        self.radioButton_2 = QRadioButton(self.groupBox_8)
        self.radioButton_2.setObjectName(u"radioButton_2")
        sizePolicy3.setHeightForWidth(self.radioButton_2.sizePolicy().hasHeightForWidth())
        self.radioButton_2.setSizePolicy(sizePolicy3)

        self.horizontalLayout_60.addWidget(self.radioButton_2)

        self.label_44 = QLabel(self.groupBox_8)
        self.label_44.setObjectName(u"label_44")

        self.horizontalLayout_60.addWidget(self.label_44)

        self.lineEdit_32 = QLineEdit(self.groupBox_8)
        self.lineEdit_32.setObjectName(u"lineEdit_32")
        sizePolicy5.setHeightForWidth(self.lineEdit_32.sizePolicy().hasHeightForWidth())
        self.lineEdit_32.setSizePolicy(sizePolicy5)
        self.lineEdit_32.setMinimumSize(QSize(20, 0))

        self.horizontalLayout_60.addWidget(self.lineEdit_32)


        self.horizontalLayout_61.addLayout(self.horizontalLayout_60)


        self.verticalLayout_28.addLayout(self.horizontalLayout_61)

        self.verticalSpacer_11 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_28.addItem(self.verticalSpacer_11)

        self.horizontalLayout_64 = QHBoxLayout()
        self.horizontalLayout_64.setObjectName(u"horizontalLayout_64")
        self.label_47 = QLabel(self.groupBox_8)
        self.label_47.setObjectName(u"label_47")

        self.horizontalLayout_64.addWidget(self.label_47)

        self.comboBox_14 = QComboBox(self.groupBox_8)
        self.comboBox_14.setObjectName(u"comboBox_14")

        self.horizontalLayout_64.addWidget(self.comboBox_14)

        self.lineEdit_34 = QLineEdit(self.groupBox_8)
        self.lineEdit_34.setObjectName(u"lineEdit_34")
        sizePolicy5.setHeightForWidth(self.lineEdit_34.sizePolicy().hasHeightForWidth())
        self.lineEdit_34.setSizePolicy(sizePolicy5)
        self.lineEdit_34.setMinimumSize(QSize(20, 0))

        self.horizontalLayout_64.addWidget(self.lineEdit_34)


        self.verticalLayout_28.addLayout(self.horizontalLayout_64)

        self.layoutWidget = QWidget(self.fit_settings)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(20, 50, 411, 220))
        self.verticalLayout_29 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_29.setObjectName(u"verticalLayout_29")
        self.verticalLayout_29.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_55 = QHBoxLayout()
        self.horizontalLayout_55.setObjectName(u"horizontalLayout_55")
        self.label_23 = QLabel(self.layoutWidget)
        self.label_23.setObjectName(u"label_23")

        self.horizontalLayout_55.addWidget(self.label_23)

        self.cb_fit_negative = QCheckBox(self.layoutWidget)
        self.cb_fit_negative.setObjectName(u"cb_fit_negative")

        self.horizontalLayout_55.addWidget(self.cb_fit_negative)


        self.verticalLayout_29.addLayout(self.horizontalLayout_55)

        self.horizontalLayout_59 = QHBoxLayout()
        self.horizontalLayout_59.setObjectName(u"horizontalLayout_59")
        self.label_25 = QLabel(self.layoutWidget)
        self.label_25.setObjectName(u"label_25")

        self.horizontalLayout_59.addWidget(self.label_25)

        self.max_iteration = QSpinBox(self.layoutWidget)
        self.max_iteration.setObjectName(u"max_iteration")
        self.max_iteration.setMaximum(10000)
        self.max_iteration.setValue(200)

        self.horizontalLayout_59.addWidget(self.max_iteration)


        self.verticalLayout_29.addLayout(self.horizontalLayout_59)

        self.horizontalLayout_63 = QHBoxLayout()
        self.horizontalLayout_63.setObjectName(u"horizontalLayout_63")
        self.label_27 = QLabel(self.layoutWidget)
        self.label_27.setObjectName(u"label_27")

        self.horizontalLayout_63.addWidget(self.label_27)

        self.cbb_fit_methods = QComboBox(self.layoutWidget)
        self.cbb_fit_methods.setObjectName(u"cbb_fit_methods")

        self.horizontalLayout_63.addWidget(self.cbb_fit_methods)


        self.verticalLayout_29.addLayout(self.horizontalLayout_63)

        self.horizontalLayout_68 = QHBoxLayout()
        self.horizontalLayout_68.setObjectName(u"horizontalLayout_68")
        self.label_55 = QLabel(self.layoutWidget)
        self.label_55.setObjectName(u"label_55")

        self.horizontalLayout_68.addWidget(self.label_55)

        self.horizontalSpacer_41 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_68.addItem(self.horizontalSpacer_41)

        self.xtol = QLineEdit(self.layoutWidget)
        self.xtol.setObjectName(u"xtol")
        self.xtol.setMaximumSize(QSize(60, 16777215))
        self.xtol.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_68.addWidget(self.xtol)


        self.verticalLayout_29.addLayout(self.horizontalLayout_68)

        self.verticalSpacer_14 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_29.addItem(self.verticalSpacer_14)

        self.btn_open_fitspy = QPushButton(self.layoutWidget)
        self.btn_open_fitspy.setObjectName(u"btn_open_fitspy")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.btn_open_fitspy.sizePolicy().hasHeightForWidth())
        self.btn_open_fitspy.setSizePolicy(sizePolicy6)
        self.btn_open_fitspy.setMinimumSize(QSize(100, 30))
        self.btn_open_fitspy.setMaximumSize(QSize(100, 30))

        self.verticalLayout_29.addWidget(self.btn_open_fitspy)

        self.label_53 = QLabel(self.fit_settings)
        self.label_53.setObjectName(u"label_53")
        self.label_53.setGeometry(QRect(10, 10, 121, 31))
        self.label_53.setFont(font1)
        self.layoutWidget1 = QWidget(self.fit_settings)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(20, 320, 715, 33))
        self.horizontalLayout_45 = QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_45.setObjectName(u"horizontalLayout_45")
        self.horizontalLayout_45.setContentsMargins(0, 0, 0, 0)
        self.btn_default_folder_model = QPushButton(self.layoutWidget1)
        self.btn_default_folder_model.setObjectName(u"btn_default_folder_model")

        self.horizontalLayout_45.addWidget(self.btn_default_folder_model)

        self.horizontalSpacer_40 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_45.addItem(self.horizontalSpacer_40)

        self.l_defaut_folder_model = QLineEdit(self.layoutWidget1)
        self.l_defaut_folder_model.setObjectName(u"l_defaut_folder_model")
        self.l_defaut_folder_model.setMinimumSize(QSize(500, 0))
        self.l_defaut_folder_model.setMaximumSize(QSize(500, 16777215))

        self.horizontalLayout_45.addWidget(self.l_defaut_folder_model)

        self.btn_refresh_model_folder = QPushButton(self.layoutWidget1)
        self.btn_refresh_model_folder.setObjectName(u"btn_refresh_model_folder")

        self.horizontalLayout_45.addWidget(self.btn_refresh_model_folder)

        self.tabWidget_2.addTab(self.fit_settings, "")

        self.verticalLayout_25.addWidget(self.tabWidget_2)

        self.splitter.addWidget(self.bottom_widget_2)

        self.gridLayout_5.addWidget(self.splitter, 1, 2, 1, 1)

        self.sidebar = QFrame(self.tab_maps)
        self.sidebar.setObjectName(u"sidebar")
        sizePolicy1.setHeightForWidth(self.sidebar.sizePolicy().hasHeightForWidth())
        self.sidebar.setSizePolicy(sizePolicy1)
        self.sidebar.setFrameShape(QFrame.StyledPanel)
        self.sidebar.setFrameShadow(QFrame.Raised)
        self.sidebar.setLineWidth(0)
        self.verticalLayout_17 = QVBoxLayout(self.sidebar)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(3, 3, 3, 3)
        self.groupBox = QGroupBox(self.sidebar)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(20)
        sizePolicy7.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy7)
        self.groupBox.setMinimumSize(QSize(290, 0))
        self.verticalLayout_2 = QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.label_64 = QLabel(self.groupBox)
        self.label_64.setObjectName(u"label_64")

        self.verticalLayout_2.addWidget(self.label_64)

        self.maps_listbox = QListWidget(self.groupBox)
        self.maps_listbox.setObjectName(u"maps_listbox")
        sizePolicy5.setHeightForWidth(self.maps_listbox.sizePolicy().hasHeightForWidth())
        self.maps_listbox.setSizePolicy(sizePolicy5)
        self.maps_listbox.setMinimumSize(QSize(270, 0))

        self.verticalLayout_2.addWidget(self.maps_listbox)

        self.horizontalLayout_29 = QHBoxLayout()
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_29.addItem(self.horizontalSpacer_4)

        self.btn_remove_wafer = QPushButton(self.groupBox)
        self.btn_remove_wafer.setObjectName(u"btn_remove_wafer")
        self.btn_remove_wafer.setMaximumSize(QSize(90, 16777215))
        self.btn_remove_wafer.setIcon(icon20)
        self.btn_remove_wafer.setIconSize(QSize(22, 22))

        self.horizontalLayout_29.addWidget(self.btn_remove_wafer)

        self.btn_view_wafer = QPushButton(self.groupBox)
        self.btn_view_wafer.setObjectName(u"btn_view_wafer")
        self.btn_view_wafer.setMaximumSize(QSize(70, 16777215))
        self.btn_view_wafer.setIcon(icon16)
        self.btn_view_wafer.setIconSize(QSize(22, 22))

        self.horizontalLayout_29.addWidget(self.btn_view_wafer)

        self.pushButton_3 = QPushButton(self.groupBox)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setMaximumSize(QSize(70, 16777215))
        self.pushButton_3.setIcon(icon17)
        self.pushButton_3.setIconSize(QSize(22, 22))

        self.horizontalLayout_29.addWidget(self.pushButton_3)


        self.verticalLayout_2.addLayout(self.horizontalLayout_29)


        self.verticalLayout_17.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(self.sidebar)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy1.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy1)
        self.groupBox_2.setMinimumSize(QSize(290, 0))
        self.verticalLayout_10 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_10.setSpacing(6)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(5, 5, 5, 5)
        self.label_58 = QLabel(self.groupBox_2)
        self.label_58.setObjectName(u"label_58")

        self.verticalLayout_10.addWidget(self.label_58)

        self.layout_quickselection = QHBoxLayout()
        self.layout_quickselection.setObjectName(u"layout_quickselection")
        self.btn_sel_all = QPushButton(self.groupBox_2)
        self.btn_sel_all.setObjectName(u"btn_sel_all")
        sizePolicy3.setHeightForWidth(self.btn_sel_all.sizePolicy().hasHeightForWidth())
        self.btn_sel_all.setSizePolicy(sizePolicy3)
        self.btn_sel_all.setMaximumSize(QSize(30, 16777215))
        self.btn_sel_all.setIcon(icon19)
        self.btn_sel_all.setIconSize(QSize(22, 22))

        self.layout_quickselection.addWidget(self.btn_sel_all)

        self.btn_sel_verti = QPushButton(self.groupBox_2)
        self.btn_sel_verti.setObjectName(u"btn_sel_verti")
        sizePolicy3.setHeightForWidth(self.btn_sel_verti.sizePolicy().hasHeightForWidth())
        self.btn_sel_verti.setSizePolicy(sizePolicy3)
        self.btn_sel_verti.setMinimumSize(QSize(30, 0))
        self.btn_sel_verti.setMaximumSize(QSize(30, 16777215))
        icon21 = QIcon()
        icon21.addFile(u":/icon/iconpack/vertical-line.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_sel_verti.setIcon(icon21)
        self.btn_sel_verti.setIconSize(QSize(22, 22))

        self.layout_quickselection.addWidget(self.btn_sel_verti)

        self.btn_sel_horiz = QPushButton(self.groupBox_2)
        self.btn_sel_horiz.setObjectName(u"btn_sel_horiz")
        sizePolicy3.setHeightForWidth(self.btn_sel_horiz.sizePolicy().hasHeightForWidth())
        self.btn_sel_horiz.setSizePolicy(sizePolicy3)
        self.btn_sel_horiz.setMinimumSize(QSize(30, 0))
        self.btn_sel_horiz.setMaximumSize(QSize(30, 16777215))
        icon22 = QIcon()
        icon22.addFile(u":/icon/iconpack/horizontal-line.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_sel_horiz.setIcon(icon22)
        self.btn_sel_horiz.setIconSize(QSize(22, 22))

        self.layout_quickselection.addWidget(self.btn_sel_horiz)

        self.btn_sel_q1 = QPushButton(self.groupBox_2)
        self.btn_sel_q1.setObjectName(u"btn_sel_q1")
        sizePolicy3.setHeightForWidth(self.btn_sel_q1.sizePolicy().hasHeightForWidth())
        self.btn_sel_q1.setSizePolicy(sizePolicy3)
        self.btn_sel_q1.setMinimumSize(QSize(30, 0))
        self.btn_sel_q1.setMaximumSize(QSize(30, 16777215))

        self.layout_quickselection.addWidget(self.btn_sel_q1)

        self.btn_sel_q2 = QPushButton(self.groupBox_2)
        self.btn_sel_q2.setObjectName(u"btn_sel_q2")
        self.btn_sel_q2.setMaximumSize(QSize(30, 16777215))

        self.layout_quickselection.addWidget(self.btn_sel_q2)

        self.btn_sel_q3 = QPushButton(self.groupBox_2)
        self.btn_sel_q3.setObjectName(u"btn_sel_q3")
        self.btn_sel_q3.setMaximumSize(QSize(30, 16777215))

        self.layout_quickselection.addWidget(self.btn_sel_q3)

        self.btn_sel_q4 = QPushButton(self.groupBox_2)
        self.btn_sel_q4.setObjectName(u"btn_sel_q4")
        self.btn_sel_q4.setMaximumSize(QSize(30, 16777215))

        self.layout_quickselection.addWidget(self.btn_sel_q4)


        self.verticalLayout_10.addLayout(self.layout_quickselection)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.checkBox_2 = QCheckBox(self.groupBox_2)
        self.checkBox_2.setObjectName(u"checkBox_2")
        self.checkBox_2.setChecked(True)

        self.horizontalLayout_8.addWidget(self.checkBox_2)

        self.horizontalSpacer_52 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_52)

        self.btn_init = QPushButton(self.groupBox_2)
        self.btn_init.setObjectName(u"btn_init")
        sizePolicy3.setHeightForWidth(self.btn_init.sizePolicy().hasHeightForWidth())
        self.btn_init.setSizePolicy(sizePolicy3)
        self.btn_init.setMinimumSize(QSize(70, 0))
        self.btn_init.setMaximumSize(QSize(70, 16777215))

        self.horizontalLayout_8.addWidget(self.btn_init)

        self.btn_show_stats = QPushButton(self.groupBox_2)
        self.btn_show_stats.setObjectName(u"btn_show_stats")
        sizePolicy3.setHeightForWidth(self.btn_show_stats.sizePolicy().hasHeightForWidth())
        self.btn_show_stats.setSizePolicy(sizePolicy3)
        self.btn_show_stats.setMaximumSize(QSize(40, 16777215))

        self.horizontalLayout_8.addWidget(self.btn_show_stats)


        self.verticalLayout_10.addLayout(self.horizontalLayout_8)

        self.listbox_layout2 = QVBoxLayout()
        self.listbox_layout2.setObjectName(u"listbox_layout2")

        self.verticalLayout_10.addLayout(self.listbox_layout2)

        self.btn_send_to_compare = QPushButton(self.groupBox_2)
        self.btn_send_to_compare.setObjectName(u"btn_send_to_compare")

        self.verticalLayout_10.addWidget(self.btn_send_to_compare)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.item_count_label = QLabel(self.groupBox_2)
        self.item_count_label.setObjectName(u"item_count_label")

        self.horizontalLayout.addWidget(self.item_count_label)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_11)


        self.verticalLayout_10.addLayout(self.horizontalLayout)


        self.verticalLayout_17.addWidget(self.groupBox_2)

        self.verticalLayout_17.setStretch(0, 30)
        self.verticalLayout_17.setStretch(1, 65)

        self.gridLayout_5.addWidget(self.sidebar, 1, 3, 1, 1)

        self.tabWidget.addTab(self.tab_maps, "")
        self.tab_graphs = QWidget()
        self.tab_graphs.setObjectName(u"tab_graphs")
        self.horizontalLayout_47 = QHBoxLayout(self.tab_graphs)
        self.horizontalLayout_47.setSpacing(5)
        self.horizontalLayout_47.setObjectName(u"horizontalLayout_47")
        self.horizontalLayout_47.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_12 = QVBoxLayout()
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.mdiArea = QMdiArea(self.tab_graphs)
        self.mdiArea.setObjectName(u"mdiArea")
        self.mdiArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdiArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.verticalLayout_12.addWidget(self.mdiArea)

        self.horizontalLayout_122 = QHBoxLayout()
        self.horizontalLayout_122.setObjectName(u"horizontalLayout_122")
        self.cbb_graph_list = QComboBox(self.tab_graphs)
        self.cbb_graph_list.setObjectName(u"cbb_graph_list")
        self.cbb_graph_list.setMinimumSize(QSize(160, 0))

        self.horizontalLayout_122.addWidget(self.cbb_graph_list)

        self.btn_minimize_all = QPushButton(self.tab_graphs)
        self.btn_minimize_all.setObjectName(u"btn_minimize_all")

        self.horizontalLayout_122.addWidget(self.btn_minimize_all)

        self.lbl_figsize = QLabel(self.tab_graphs)
        self.lbl_figsize.setObjectName(u"lbl_figsize")

        self.horizontalLayout_122.addWidget(self.lbl_figsize)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_122.addItem(self.horizontalSpacer_2)

        self.horizontalLayout_123 = QHBoxLayout()
        self.horizontalLayout_123.setObjectName(u"horizontalLayout_123")
        self.horizontalLayout_123.setContentsMargins(5, -1, 5, -1)
        self.label_16 = QLabel(self.tab_graphs)
        self.label_16.setObjectName(u"label_16")

        self.horizontalLayout_123.addWidget(self.label_16)

        self.spb_dpi = QDoubleSpinBox(self.tab_graphs)
        self.spb_dpi.setObjectName(u"spb_dpi")
        self.spb_dpi.setDecimals(0)
        self.spb_dpi.setMinimum(100.000000000000000)
        self.spb_dpi.setMaximum(300.000000000000000)
        self.spb_dpi.setSingleStep(10.000000000000000)
        self.spb_dpi.setValue(110.000000000000000)

        self.horizontalLayout_123.addWidget(self.spb_dpi)


        self.horizontalLayout_122.addLayout(self.horizontalLayout_123)

        self.horizontalLayout_110 = QHBoxLayout()
        self.horizontalLayout_110.setObjectName(u"horizontalLayout_110")
        self.horizontalLayout_110.setContentsMargins(5, 0, 5, -1)
        self.label_92 = QLabel(self.tab_graphs)
        self.label_92.setObjectName(u"label_92")

        self.horizontalLayout_110.addWidget(self.label_92)

        self.x_rot = QDoubleSpinBox(self.tab_graphs)
        self.x_rot.setObjectName(u"x_rot")
        self.x_rot.setDecimals(0)
        self.x_rot.setMaximum(50.000000000000000)
        self.x_rot.setSingleStep(10.000000000000000)

        self.horizontalLayout_110.addWidget(self.x_rot)


        self.horizontalLayout_122.addLayout(self.horizontalLayout_110)

        self.cb_legend_outside = QCheckBox(self.tab_graphs)
        self.cb_legend_outside.setObjectName(u"cb_legend_outside")

        self.horizontalLayout_122.addWidget(self.cb_legend_outside)

        self.cb_grid = QCheckBox(self.tab_graphs)
        self.cb_grid.setObjectName(u"cb_grid")

        self.horizontalLayout_122.addWidget(self.cb_grid)

        self.btn_copy_graph = QPushButton(self.tab_graphs)
        self.btn_copy_graph.setObjectName(u"btn_copy_graph")
        self.btn_copy_graph.setIcon(icon11)

        self.horizontalLayout_122.addWidget(self.btn_copy_graph)


        self.verticalLayout_12.addLayout(self.horizontalLayout_122)


        self.horizontalLayout_47.addLayout(self.verticalLayout_12)

        self.sidebar_2 = QFrame(self.tab_graphs)
        self.sidebar_2.setObjectName(u"sidebar_2")
        self.sidebar_2.setMinimumSize(QSize(390, 0))
        self.sidebar_2.setMaximumSize(QSize(390, 16777215))
        self.sidebar_2.setFrameShape(QFrame.StyledPanel)
        self.sidebar_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout_23 = QVBoxLayout(self.sidebar_2)
        self.verticalLayout_23.setSpacing(5)
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.verticalLayout_23.setContentsMargins(2, 2, 2, 2)
        self.groupBox_loaded_df_2 = QGroupBox(self.sidebar_2)
        self.groupBox_loaded_df_2.setObjectName(u"groupBox_loaded_df_2")
        self.verticalLayout_51 = QVBoxLayout(self.groupBox_loaded_df_2)
        self.verticalLayout_51.setSpacing(5)
        self.verticalLayout_51.setObjectName(u"verticalLayout_51")
        self.verticalLayout_51.setContentsMargins(2, 2, 2, 2)
        self.horizontalLayout_48 = QHBoxLayout()
        self.horizontalLayout_48.setSpacing(2)
        self.horizontalLayout_48.setObjectName(u"horizontalLayout_48")
        self.dfs_listbox = QListWidget(self.groupBox_loaded_df_2)
        self.dfs_listbox.setObjectName(u"dfs_listbox")
        self.dfs_listbox.setMinimumSize(QSize(0, 70))
        self.dfs_listbox.setMaximumSize(QSize(16777215, 70))

        self.horizontalLayout_48.addWidget(self.dfs_listbox)


        self.verticalLayout_51.addLayout(self.horizontalLayout_48)

        self.horizontalLayout_109 = QHBoxLayout()
        self.horizontalLayout_109.setSpacing(5)
        self.horizontalLayout_109.setObjectName(u"horizontalLayout_109")
        self.horizontalSpacer_44 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_109.addItem(self.horizontalSpacer_44)

        self.merge_dfs_2 = QPushButton(self.groupBox_loaded_df_2)
        self.merge_dfs_2.setObjectName(u"merge_dfs_2")
        sizePolicy5.setHeightForWidth(self.merge_dfs_2.sizePolicy().hasHeightForWidth())
        self.merge_dfs_2.setSizePolicy(sizePolicy5)
        self.merge_dfs_2.setMinimumSize(QSize(60, 30))
        self.merge_dfs_2.setMaximumSize(QSize(60, 30))

        self.horizontalLayout_109.addWidget(self.merge_dfs_2)

        self.btn_view_df_3 = QPushButton(self.groupBox_loaded_df_2)
        self.btn_view_df_3.setObjectName(u"btn_view_df_3")
        self.btn_view_df_3.setMinimumSize(QSize(30, 30))
        self.btn_view_df_3.setMaximumSize(QSize(30, 30))
        self.btn_view_df_3.setAutoFillBackground(True)
        self.btn_view_df_3.setIcon(icon16)
        self.btn_view_df_3.setIconSize(QSize(25, 25))

        self.horizontalLayout_109.addWidget(self.btn_view_df_3)

        self.btn_remove_df_2 = QPushButton(self.groupBox_loaded_df_2)
        self.btn_remove_df_2.setObjectName(u"btn_remove_df_2")
        self.btn_remove_df_2.setMinimumSize(QSize(30, 30))
        self.btn_remove_df_2.setMaximumSize(QSize(30, 30))
        self.btn_remove_df_2.setIcon(icon20)
        self.btn_remove_df_2.setIconSize(QSize(25, 25))

        self.horizontalLayout_109.addWidget(self.btn_remove_df_2)

        self.btn_save_df_2 = QPushButton(self.groupBox_loaded_df_2)
        self.btn_save_df_2.setObjectName(u"btn_save_df_2")
        self.btn_save_df_2.setMinimumSize(QSize(30, 30))
        self.btn_save_df_2.setMaximumSize(QSize(30, 30))
        self.btn_save_df_2.setIcon(icon17)
        self.btn_save_df_2.setIconSize(QSize(25, 25))

        self.horizontalLayout_109.addWidget(self.btn_save_df_2)


        self.verticalLayout_51.addLayout(self.horizontalLayout_109)


        self.verticalLayout_23.addWidget(self.groupBox_loaded_df_2)

        self.groupBox_df_manip_5 = QGroupBox(self.sidebar_2)
        self.groupBox_df_manip_5.setObjectName(u"groupBox_df_manip_5")
        self.groupBox_df_manip_5.setMinimumSize(QSize(0, 150))
        self.verticalLayout_56 = QVBoxLayout(self.groupBox_df_manip_5)
        self.verticalLayout_56.setSpacing(5)
        self.verticalLayout_56.setObjectName(u"verticalLayout_56")
        self.verticalLayout_56.setContentsMargins(2, 2, 2, 2)
        self.horizontalLayout_99 = QHBoxLayout()
        self.horizontalLayout_99.setSpacing(5)
        self.horizontalLayout_99.setObjectName(u"horizontalLayout_99")
        self.filter_query = QLineEdit(self.groupBox_df_manip_5)
        self.filter_query.setObjectName(u"filter_query")

        self.horizontalLayout_99.addWidget(self.filter_query)

        self.btn_add_filter_4 = QPushButton(self.groupBox_df_manip_5)
        self.btn_add_filter_4.setObjectName(u"btn_add_filter_4")
        icon23 = QIcon()
        icon23.addFile(u":/icon/iconpack/add.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_add_filter_4.setIcon(icon23)

        self.horizontalLayout_99.addWidget(self.btn_add_filter_4)

        self.btn_remove_filters_4 = QPushButton(self.groupBox_df_manip_5)
        self.btn_remove_filters_4.setObjectName(u"btn_remove_filters_4")
        icon24 = QIcon()
        icon24.addFile(u":/icon/iconpack/close.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_remove_filters_4.setIcon(icon24)

        self.horizontalLayout_99.addWidget(self.btn_remove_filters_4)

        self.btn_apply_filters_4 = QPushButton(self.groupBox_df_manip_5)
        self.btn_apply_filters_4.setObjectName(u"btn_apply_filters_4")
        icon25 = QIcon()
        icon25.addFile(u":/icon/iconpack/done.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_apply_filters_4.setIcon(icon25)

        self.horizontalLayout_99.addWidget(self.btn_apply_filters_4)


        self.verticalLayout_56.addLayout(self.horizontalLayout_99)

        self.listbox_filters = QListWidget(self.groupBox_df_manip_5)
        self.listbox_filters.setObjectName(u"listbox_filters")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.listbox_filters.sizePolicy().hasHeightForWidth())
        self.listbox_filters.setSizePolicy(sizePolicy8)
        self.listbox_filters.setMinimumSize(QSize(0, 80))
        self.listbox_filters.setMaximumSize(QSize(16777215, 100))

        self.verticalLayout_56.addWidget(self.listbox_filters)


        self.verticalLayout_23.addWidget(self.groupBox_df_manip_5)

        self.horizontalLayout_114 = QHBoxLayout()
        self.horizontalLayout_114.setObjectName(u"horizontalLayout_114")
        self.horizontalLayout_114.setContentsMargins(10, 5, 10, 5)
        self.btn_add_graph = QPushButton(self.sidebar_2)
        self.btn_add_graph.setObjectName(u"btn_add_graph")
        self.btn_add_graph.setMinimumSize(QSize(100, 0))
        self.btn_add_graph.setMaximumSize(QSize(100, 100))
        icon26 = QIcon()
        icon26.addFile(u":/icon/iconpack/icons8-graph-96.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_add_graph.setIcon(icon26)
        self.btn_add_graph.setIconSize(QSize(30, 30))

        self.horizontalLayout_114.addWidget(self.btn_add_graph)

        self.horizontalSpacer_49 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_114.addItem(self.horizontalSpacer_49)

        self.btn_upd_graph = QPushButton(self.sidebar_2)
        self.btn_upd_graph.setObjectName(u"btn_upd_graph")
        self.btn_upd_graph.setMinimumSize(QSize(110, 0))
        self.btn_upd_graph.setMaximumSize(QSize(110, 100))
        icon27 = QIcon()
        icon27.addFile(u":/icon/iconpack/icons8-update-100.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_upd_graph.setIcon(icon27)
        self.btn_upd_graph.setIconSize(QSize(20, 20))

        self.horizontalLayout_114.addWidget(self.btn_upd_graph)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_114.addItem(self.horizontalSpacer)

        self.btn_add_line = QPushButton(self.sidebar_2)
        self.btn_add_line.setObjectName(u"btn_add_line")
        self.btn_add_line.setMinimumSize(QSize(70, 0))
        self.btn_add_line.setMaximumSize(QSize(70, 16777215))

        self.horizontalLayout_114.addWidget(self.btn_add_line)


        self.verticalLayout_23.addLayout(self.horizontalLayout_114)

        self.tabWidget_4 = QTabWidget(self.sidebar_2)
        self.tabWidget_4.setObjectName(u"tabWidget_4")
        self.tabWidget_4.setMinimumSize(QSize(0, 400))
        self.tab_plot_settings = QWidget()
        self.tab_plot_settings.setObjectName(u"tab_plot_settings")
        self.verticalLayout = QVBoxLayout(self.tab_plot_settings)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.scrollArea = QScrollArea(self.tab_plot_settings)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 395, 454))
        self.verticalLayout_3 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_115 = QHBoxLayout()
        self.horizontalLayout_115.setObjectName(u"horizontalLayout_115")
        self.label_96 = QLabel(self.scrollAreaWidgetContents)
        self.label_96.setObjectName(u"label_96")
        self.label_96.setMinimumSize(QSize(80, 0))
        self.label_96.setMaximumSize(QSize(80, 16777215))
        self.label_96.setFont(font)

        self.horizontalLayout_115.addWidget(self.label_96)

        self.cbb_plotstyle = QComboBox(self.scrollAreaWidgetContents)
        self.cbb_plotstyle.setObjectName(u"cbb_plotstyle")

        self.horizontalLayout_115.addWidget(self.cbb_plotstyle)

        self.label_93 = QLabel(self.scrollAreaWidgetContents)
        self.label_93.setObjectName(u"label_93")

        self.horizontalLayout_115.addWidget(self.label_93)

        self.cbb_palette = QComboBox(self.scrollAreaWidgetContents)
        self.cbb_palette.setObjectName(u"cbb_palette")

        self.horizontalLayout_115.addWidget(self.cbb_palette)


        self.verticalLayout_3.addLayout(self.horizontalLayout_115)

        self.horizontalLayout_71 = QHBoxLayout()
        self.horizontalLayout_71.setObjectName(u"horizontalLayout_71")
        self.label_82 = QLabel(self.scrollAreaWidgetContents)
        self.label_82.setObjectName(u"label_82")
        self.label_82.setMinimumSize(QSize(30, 0))
        self.label_82.setMaximumSize(QSize(30, 16777215))

        self.horizontalLayout_71.addWidget(self.label_82)

        self.cbb_x_2 = QComboBox(self.scrollAreaWidgetContents)
        self.cbb_x_2.setObjectName(u"cbb_x_2")

        self.horizontalLayout_71.addWidget(self.cbb_x_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout_71)

        self.horizontalLayout_88 = QHBoxLayout()
        self.horizontalLayout_88.setObjectName(u"horizontalLayout_88")
        self.label_84 = QLabel(self.scrollAreaWidgetContents)
        self.label_84.setObjectName(u"label_84")
        self.label_84.setMinimumSize(QSize(30, 0))
        self.label_84.setMaximumSize(QSize(30, 16777215))

        self.horizontalLayout_88.addWidget(self.label_84)

        self.cbb_y_2 = QComboBox(self.scrollAreaWidgetContents)
        self.cbb_y_2.setObjectName(u"cbb_y_2")

        self.horizontalLayout_88.addWidget(self.cbb_y_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout_88)

        self.horizontalLayout_89 = QHBoxLayout()
        self.horizontalLayout_89.setObjectName(u"horizontalLayout_89")
        self.label_85 = QLabel(self.scrollAreaWidgetContents)
        self.label_85.setObjectName(u"label_85")
        self.label_85.setMinimumSize(QSize(30, 0))
        self.label_85.setMaximumSize(QSize(30, 16777215))

        self.horizontalLayout_89.addWidget(self.label_85)

        self.cbb_z_2 = QComboBox(self.scrollAreaWidgetContents)
        self.cbb_z_2.setObjectName(u"cbb_z_2")

        self.horizontalLayout_89.addWidget(self.cbb_z_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout_89)

        self.label = QLabel(self.scrollAreaWidgetContents)
        self.label.setObjectName(u"label")

        self.verticalLayout_3.addWidget(self.label)

        self.horizontalLayout_107 = QHBoxLayout()
        self.horizontalLayout_107.setObjectName(u"horizontalLayout_107")
        self.label_91 = QLabel(self.scrollAreaWidgetContents)
        self.label_91.setObjectName(u"label_91")
        self.label_91.setFont(font)

        self.horizontalLayout_107.addWidget(self.label_91)

        self.lbl_plot_title = QLineEdit(self.scrollAreaWidgetContents)
        self.lbl_plot_title.setObjectName(u"lbl_plot_title")

        self.horizontalLayout_107.addWidget(self.lbl_plot_title)


        self.verticalLayout_3.addLayout(self.horizontalLayout_107)

        self.horizontalLayout_91 = QHBoxLayout()
        self.horizontalLayout_91.setObjectName(u"horizontalLayout_91")
        self.label_86 = QLabel(self.scrollAreaWidgetContents)
        self.label_86.setObjectName(u"label_86")

        self.horizontalLayout_91.addWidget(self.label_86)

        self.lbl_xlabel = QLineEdit(self.scrollAreaWidgetContents)
        self.lbl_xlabel.setObjectName(u"lbl_xlabel")

        self.horizontalLayout_91.addWidget(self.lbl_xlabel)


        self.verticalLayout_3.addLayout(self.horizontalLayout_91)

        self.horizontalLayout_92 = QHBoxLayout()
        self.horizontalLayout_92.setObjectName(u"horizontalLayout_92")
        self.label_87 = QLabel(self.scrollAreaWidgetContents)
        self.label_87.setObjectName(u"label_87")

        self.horizontalLayout_92.addWidget(self.label_87)

        self.lbl_ylabel = QLineEdit(self.scrollAreaWidgetContents)
        self.lbl_ylabel.setObjectName(u"lbl_ylabel")

        self.horizontalLayout_92.addWidget(self.lbl_ylabel)


        self.verticalLayout_3.addLayout(self.horizontalLayout_92)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_97 = QLabel(self.scrollAreaWidgetContents)
        self.label_97.setObjectName(u"label_97")
        self.label_97.setFont(font)

        self.horizontalLayout_19.addWidget(self.label_97)

        self.btn_get_limits = QPushButton(self.scrollAreaWidgetContents)
        self.btn_get_limits.setObjectName(u"btn_get_limits")

        self.horizontalLayout_19.addWidget(self.btn_get_limits)

        self.btn_clear_limits = QPushButton(self.scrollAreaWidgetContents)
        self.btn_clear_limits.setObjectName(u"btn_clear_limits")

        self.horizontalLayout_19.addWidget(self.btn_clear_limits)


        self.verticalLayout_3.addLayout(self.horizontalLayout_19)

        self.horizontalLayout_102 = QHBoxLayout()
        self.horizontalLayout_102.setObjectName(u"horizontalLayout_102")
        self.label_89 = QLabel(self.scrollAreaWidgetContents)
        self.label_89.setObjectName(u"label_89")

        self.horizontalLayout_102.addWidget(self.label_89)

        self.xmin_2 = QLineEdit(self.scrollAreaWidgetContents)
        self.xmin_2.setObjectName(u"xmin_2")

        self.horizontalLayout_102.addWidget(self.xmin_2)

        self.xmax_2 = QLineEdit(self.scrollAreaWidgetContents)
        self.xmax_2.setObjectName(u"xmax_2")

        self.horizontalLayout_102.addWidget(self.xmax_2)

        self.label_90 = QLabel(self.scrollAreaWidgetContents)
        self.label_90.setObjectName(u"label_90")

        self.horizontalLayout_102.addWidget(self.label_90)

        self.ymin_2 = QLineEdit(self.scrollAreaWidgetContents)
        self.ymin_2.setObjectName(u"ymin_2")

        self.horizontalLayout_102.addWidget(self.ymin_2)

        self.ymax_2 = QLineEdit(self.scrollAreaWidgetContents)
        self.ymax_2.setObjectName(u"ymax_2")

        self.horizontalLayout_102.addWidget(self.ymax_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout_102)

        self.label_2 = QLabel(self.scrollAreaWidgetContents)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_3.addWidget(self.label_2)

        self.horizontalLayout_93 = QHBoxLayout()
        self.horizontalLayout_93.setObjectName(u"horizontalLayout_93")
        self.label_88 = QLabel(self.scrollAreaWidgetContents)
        self.label_88.setObjectName(u"label_88")

        self.horizontalLayout_93.addWidget(self.label_88)

        self.lbl_zlabel = QLineEdit(self.scrollAreaWidgetContents)
        self.lbl_zlabel.setObjectName(u"lbl_zlabel")

        self.horizontalLayout_93.addWidget(self.lbl_zlabel)


        self.verticalLayout_3.addLayout(self.horizontalLayout_93)

        self.horizontalLayout_113 = QHBoxLayout()
        self.horizontalLayout_113.setObjectName(u"horizontalLayout_113")
        self.label_94 = QLabel(self.scrollAreaWidgetContents)
        self.label_94.setObjectName(u"label_94")

        self.horizontalLayout_113.addWidget(self.label_94)

        self.zmin_2 = QLineEdit(self.scrollAreaWidgetContents)
        self.zmin_2.setObjectName(u"zmin_2")

        self.horizontalLayout_113.addWidget(self.zmin_2)

        self.zmax_2 = QLineEdit(self.scrollAreaWidgetContents)
        self.zmax_2.setObjectName(u"zmax_2")

        self.horizontalLayout_113.addWidget(self.zmax_2)

        self.horizontalSpacer_43 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_113.addItem(self.horizontalSpacer_43)


        self.verticalLayout_3.addLayout(self.horizontalLayout_113)

        self.label_3 = QLabel(self.scrollAreaWidgetContents)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_3.addWidget(self.label_3)

        self.horizontalLayout_124 = QHBoxLayout()
        self.horizontalLayout_124.setObjectName(u"horizontalLayout_124")
        self.label_98 = QLabel(self.scrollAreaWidgetContents)
        self.label_98.setObjectName(u"label_98")

        self.horizontalLayout_124.addWidget(self.label_98)

        self.lbl_wafersize = QLineEdit(self.scrollAreaWidgetContents)
        self.lbl_wafersize.setObjectName(u"lbl_wafersize")

        self.horizontalLayout_124.addWidget(self.lbl_wafersize)


        self.verticalLayout_3.addLayout(self.horizontalLayout_124)

        self.verticalSpacer_8 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_8)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout.addWidget(self.scrollArea)

        self.tabWidget_4.addTab(self.tab_plot_settings, "")
        self.tab_more_options = QWidget()
        self.tab_more_options.setObjectName(u"tab_more_options")
        self.verticalLayout_4 = QVBoxLayout(self.tab_more_options)
        self.verticalLayout_4.setSpacing(2)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(2, 2, 2, 2)
        self.scrollArea_8 = QScrollArea(self.tab_more_options)
        self.scrollArea_8.setObjectName(u"scrollArea_8")
        self.scrollArea_8.setWidgetResizable(True)
        self.scrollAreaWidgetContents_8 = QWidget()
        self.scrollAreaWidgetContents_8.setObjectName(u"scrollAreaWidgetContents_8")
        self.scrollAreaWidgetContents_8.setGeometry(QRect(0, 0, 277, 298))
        self.verticalLayout_11 = QVBoxLayout(self.scrollAreaWidgetContents_8)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.cb_legend_visible = QCheckBox(self.scrollAreaWidgetContents_8)
        self.cb_legend_visible.setObjectName(u"cb_legend_visible")
        self.cb_legend_visible.setChecked(True)

        self.verticalLayout_11.addWidget(self.cb_legend_visible)

        self.cb_show_err_bar_plot = QCheckBox(self.scrollAreaWidgetContents_8)
        self.cb_show_err_bar_plot.setObjectName(u"cb_show_err_bar_plot")
        self.cb_show_err_bar_plot.setChecked(True)

        self.verticalLayout_11.addWidget(self.cb_show_err_bar_plot)

        self.cb_wafer_stats = QCheckBox(self.scrollAreaWidgetContents_8)
        self.cb_wafer_stats.setObjectName(u"cb_wafer_stats")
        self.cb_wafer_stats.setChecked(True)

        self.verticalLayout_11.addWidget(self.cb_wafer_stats)

        self.cb_join_for_point_plot = QCheckBox(self.scrollAreaWidgetContents_8)
        self.cb_join_for_point_plot.setObjectName(u"cb_join_for_point_plot")

        self.verticalLayout_11.addWidget(self.cb_join_for_point_plot)

        self.horizontalLayout_126 = QHBoxLayout()
        self.horizontalLayout_126.setObjectName(u"horizontalLayout_126")
        self.cb_trendline_eq = QCheckBox(self.scrollAreaWidgetContents_8)
        self.cb_trendline_eq.setObjectName(u"cb_trendline_eq")
        self.cb_trendline_eq.setChecked(True)

        self.horizontalLayout_126.addWidget(self.cb_trendline_eq)

        self.spb_trendline_oder = QDoubleSpinBox(self.scrollAreaWidgetContents_8)
        self.spb_trendline_oder.setObjectName(u"spb_trendline_oder")
        self.spb_trendline_oder.setDecimals(0)
        self.spb_trendline_oder.setMinimum(1.000000000000000)
        self.spb_trendline_oder.setMaximum(10.000000000000000)

        self.horizontalLayout_126.addWidget(self.spb_trendline_oder)

        self.label_18 = QLabel(self.scrollAreaWidgetContents_8)
        self.label_18.setObjectName(u"label_18")

        self.horizontalLayout_126.addWidget(self.label_18)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_126.addItem(self.horizontalSpacer_5)


        self.verticalLayout_11.addLayout(self.horizontalLayout_126)

        self.line_3 = QFrame(self.scrollAreaWidgetContents_8)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_11.addWidget(self.line_3)

        self.legends_loc = QHBoxLayout()
        self.legends_loc.setObjectName(u"legends_loc")
        self.label_17 = QLabel(self.scrollAreaWidgetContents_8)
        self.label_17.setObjectName(u"label_17")

        self.legends_loc.addWidget(self.label_17)

        self.cbb_legend_loc = QComboBox(self.scrollAreaWidgetContents_8)
        self.cbb_legend_loc.setObjectName(u"cbb_legend_loc")

        self.legends_loc.addWidget(self.cbb_legend_loc)

        self.horizontalSpacer_35 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.legends_loc.addItem(self.horizontalSpacer_35)


        self.verticalLayout_11.addLayout(self.legends_loc)

        self.line = QFrame(self.scrollAreaWidgetContents_8)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_11.addWidget(self.line)

        self.label_13 = QLabel(self.scrollAreaWidgetContents_8)
        self.label_13.setObjectName(u"label_13")
        font2 = QFont()
        font2.setPointSize(9)
        font2.setBold(True)
        self.label_13.setFont(font2)
        self.label_13.setAlignment(Qt.AlignCenter)

        self.verticalLayout_11.addWidget(self.label_13)

        self.main_layout = QHBoxLayout()
        self.main_layout.setObjectName(u"main_layout")

        self.verticalLayout_11.addLayout(self.main_layout)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalSpacer_48 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_13.addItem(self.horizontalSpacer_48)


        self.verticalLayout_11.addLayout(self.horizontalLayout_13)

        self.line_2 = QFrame(self.scrollAreaWidgetContents_8)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_11.addWidget(self.line_2)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_11.addItem(self.verticalSpacer_3)

        self.scrollArea_8.setWidget(self.scrollAreaWidgetContents_8)

        self.verticalLayout_4.addWidget(self.scrollArea_8)

        self.tabWidget_4.addTab(self.tab_more_options, "")
        self.tab_multi_axes = QWidget()
        self.tab_multi_axes.setObjectName(u"tab_multi_axes")
        self.verticalLayout_8 = QVBoxLayout(self.tab_multi_axes)
        self.verticalLayout_8.setSpacing(2)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(2, 2, 2, 2)
        self.scrollArea_10 = QScrollArea(self.tab_multi_axes)
        self.scrollArea_10.setObjectName(u"scrollArea_10")
        self.scrollArea_10.setWidgetResizable(True)
        self.scrollAreaWidgetContents_10 = QWidget()
        self.scrollAreaWidgetContents_10.setObjectName(u"scrollAreaWidgetContents_10")
        self.scrollAreaWidgetContents_10.setGeometry(QRect(0, 0, 364, 371))
        self.verticalLayout_9 = QVBoxLayout(self.scrollAreaWidgetContents_10)
        self.verticalLayout_9.setSpacing(5)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(5, 5, 5, 5)
        self.label_12 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_12.setObjectName(u"label_12")

        self.verticalLayout_9.addWidget(self.label_12)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label100 = QLabel(self.scrollAreaWidgetContents_10)
        self.label100.setObjectName(u"label100")
        self.label100.setMaximumSize(QSize(50, 16777215))

        self.horizontalLayout_6.addWidget(self.label100)

        self.cbb_y2_2 = QComboBox(self.scrollAreaWidgetContents_10)
        self.cbb_y2_2.setObjectName(u"cbb_y2_2")
        self.cbb_y2_2.setMinimumSize(QSize(160, 0))

        self.horizontalLayout_6.addWidget(self.cbb_y2_2)

        self.btn_add_y2 = QPushButton(self.scrollAreaWidgetContents_10)
        self.btn_add_y2.setObjectName(u"btn_add_y2")

        self.horizontalLayout_6.addWidget(self.btn_add_y2)

        self.btn_remove_y2 = QPushButton(self.scrollAreaWidgetContents_10)
        self.btn_remove_y2.setObjectName(u"btn_remove_y2")

        self.horizontalLayout_6.addWidget(self.btn_remove_y2)


        self.verticalLayout_9.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_4 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_2.addWidget(self.label_4)

        self.lbl_y2label = QLineEdit(self.scrollAreaWidgetContents_10)
        self.lbl_y2label.setObjectName(u"lbl_y2label")

        self.horizontalLayout_2.addWidget(self.lbl_y2label)


        self.verticalLayout_9.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_5 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_3.addWidget(self.label_5)

        self.y2min_2 = QLineEdit(self.scrollAreaWidgetContents_10)
        self.y2min_2.setObjectName(u"y2min_2")

        self.horizontalLayout_3.addWidget(self.y2min_2)

        self.label_6 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_3.addWidget(self.label_6)

        self.y2max_2 = QLineEdit(self.scrollAreaWidgetContents_10)
        self.y2max_2.setObjectName(u"y2max_2")

        self.horizontalLayout_3.addWidget(self.y2max_2)


        self.verticalLayout_9.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_11 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setMaximumSize(QSize(50, 16777215))

        self.horizontalLayout_10.addWidget(self.label_11)

        self.cbb_y3_2 = QComboBox(self.scrollAreaWidgetContents_10)
        self.cbb_y3_2.setObjectName(u"cbb_y3_2")
        self.cbb_y3_2.setMinimumSize(QSize(160, 0))

        self.horizontalLayout_10.addWidget(self.cbb_y3_2)

        self.btn_add_y3 = QPushButton(self.scrollAreaWidgetContents_10)
        self.btn_add_y3.setObjectName(u"btn_add_y3")

        self.horizontalLayout_10.addWidget(self.btn_add_y3)

        self.btn_remove_y3 = QPushButton(self.scrollAreaWidgetContents_10)
        self.btn_remove_y3.setObjectName(u"btn_remove_y3")

        self.horizontalLayout_10.addWidget(self.btn_remove_y3)


        self.verticalLayout_9.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_7 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_4.addWidget(self.label_7)

        self.lbl_y3label = QLineEdit(self.scrollAreaWidgetContents_10)
        self.lbl_y3label.setObjectName(u"lbl_y3label")

        self.horizontalLayout_4.addWidget(self.lbl_y3label)


        self.verticalLayout_9.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_9 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_5.addWidget(self.label_9)

        self.y3min_2 = QLineEdit(self.scrollAreaWidgetContents_10)
        self.y3min_2.setObjectName(u"y3min_2")

        self.horizontalLayout_5.addWidget(self.y3min_2)

        self.label_8 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_5.addWidget(self.label_8)

        self.y3max_2 = QLineEdit(self.scrollAreaWidgetContents_10)
        self.y3max_2.setObjectName(u"y3max_2")

        self.horizontalLayout_5.addWidget(self.y3max_2)


        self.verticalLayout_9.addLayout(self.horizontalLayout_5)

        self.verticalSpacer_20 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_9.addItem(self.verticalSpacer_20)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label100_2 = QLabel(self.scrollAreaWidgetContents_10)
        self.label100_2.setObjectName(u"label100_2")
        self.label100_2.setMaximumSize(QSize(50, 16777215))

        self.horizontalLayout_11.addWidget(self.label100_2)

        self.cbb_y12 = QComboBox(self.scrollAreaWidgetContents_10)
        self.cbb_y12.setObjectName(u"cbb_y12")
        self.cbb_y12.setMinimumSize(QSize(160, 0))

        self.horizontalLayout_11.addWidget(self.cbb_y12)

        self.btn_add_y12 = QPushButton(self.scrollAreaWidgetContents_10)
        self.btn_add_y12.setObjectName(u"btn_add_y12")

        self.horizontalLayout_11.addWidget(self.btn_add_y12)

        self.btn_remove_y12 = QPushButton(self.scrollAreaWidgetContents_10)
        self.btn_remove_y12.setObjectName(u"btn_remove_y12")

        self.horizontalLayout_11.addWidget(self.btn_remove_y12)


        self.verticalLayout_9.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_10 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_12.addWidget(self.label_10)

        self.lbl_y12label = QLineEdit(self.scrollAreaWidgetContents_10)
        self.lbl_y12label.setObjectName(u"lbl_y12label")

        self.horizontalLayout_12.addWidget(self.lbl_y12label)


        self.verticalLayout_9.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.label_14 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setMaximumSize(QSize(50, 16777215))

        self.horizontalLayout_14.addWidget(self.label_14)

        self.cbb_y13 = QComboBox(self.scrollAreaWidgetContents_10)
        self.cbb_y13.setObjectName(u"cbb_y13")
        self.cbb_y13.setMinimumSize(QSize(160, 0))

        self.horizontalLayout_14.addWidget(self.cbb_y13)

        self.btn_add_y13 = QPushButton(self.scrollAreaWidgetContents_10)
        self.btn_add_y13.setObjectName(u"btn_add_y13")

        self.horizontalLayout_14.addWidget(self.btn_add_y13)

        self.btn_remove_y13 = QPushButton(self.scrollAreaWidgetContents_10)
        self.btn_remove_y13.setObjectName(u"btn_remove_y13")

        self.horizontalLayout_14.addWidget(self.btn_remove_y13)


        self.verticalLayout_9.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.label_15 = QLabel(self.scrollAreaWidgetContents_10)
        self.label_15.setObjectName(u"label_15")

        self.horizontalLayout_15.addWidget(self.label_15)

        self.lbl_y3label_2 = QLineEdit(self.scrollAreaWidgetContents_10)
        self.lbl_y3label_2.setObjectName(u"lbl_y3label_2")

        self.horizontalLayout_15.addWidget(self.lbl_y3label_2)


        self.verticalLayout_9.addLayout(self.horizontalLayout_15)

        self.scrollArea_10.setWidget(self.scrollAreaWidgetContents_10)

        self.verticalLayout_8.addWidget(self.scrollArea_10)

        self.tabWidget_4.addTab(self.tab_multi_axes, "")

        self.verticalLayout_23.addWidget(self.tabWidget_4)

        self.widget = QWidget(self.sidebar_2)
        self.widget.setObjectName(u"widget")
        self.gridLayout_2 = QGridLayout(self.widget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")

        self.verticalLayout_23.addWidget(self.widget)

        self.verticalLayout_23.setStretch(0, 10)
        self.verticalLayout_23.setStretch(1, 10)
        self.verticalLayout_23.setStretch(2, 5)
        self.verticalLayout_23.setStretch(3, 74)
        self.verticalLayout_23.setStretch(4, 1)

        self.horizontalLayout_47.addWidget(self.sidebar_2)

        self.tabWidget.addTab(self.tab_graphs, "")

        self.verticalLayout_15.addWidget(self.tabWidget)

        self.layout_statusbar = QHBoxLayout()
        self.layout_statusbar.setObjectName(u"layout_statusbar")
        self.layout_statusbar.setContentsMargins(5, 5, 5, 5)
        self.label_38 = QLabel(self.centralwidget)
        self.label_38.setObjectName(u"label_38")

        self.layout_statusbar.addWidget(self.label_38)

        self.label_19 = QLabel(self.centralwidget)
        self.label_19.setObjectName(u"label_19")

        self.layout_statusbar.addWidget(self.label_19)

        self.label_21 = QLabel(self.centralwidget)
        self.label_21.setObjectName(u"label_21")

        self.layout_statusbar.addWidget(self.label_21)

        self.ncpus = QSpinBox(self.centralwidget)
        self.ncpus.setObjectName(u"ncpus")
        self.ncpus.setMinimum(1)
        self.ncpus.setMaximum(64)

        self.layout_statusbar.addWidget(self.ncpus)

        self.horizontalSpacer_58 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.layout_statusbar.addItem(self.horizontalSpacer_58)

        self.progressText = QLabel(self.centralwidget)
        self.progressText.setObjectName(u"progressText")

        self.layout_statusbar.addWidget(self.progressText)

        self.label_95 = QLabel(self.centralwidget)
        self.label_95.setObjectName(u"label_95")

        self.layout_statusbar.addWidget(self.label_95)

        self.progressBar = QProgressBar(self.centralwidget)
        self.progressBar.setObjectName(u"progressBar")
        sizePolicy5.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy5)
        self.progressBar.setMinimumSize(QSize(200, 10))
        self.progressBar.setMaximumSize(QSize(200, 10))
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(100)
        self.progressBar.setTextVisible(True)
        self.progressBar.setInvertedAppearance(False)

        self.layout_statusbar.addWidget(self.progressBar)

        self.label_24 = QLabel(self.centralwidget)
        self.label_24.setObjectName(u"label_24")

        self.layout_statusbar.addWidget(self.label_24)


        self.verticalLayout_15.addLayout(self.layout_statusbar)

        mainWindow.setCentralWidget(self.centralwidget)
        self.toolBar = QToolBar(mainWindow)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setMinimumSize(QSize(0, 0))
        self.toolBar.setMaximumSize(QSize(16777215, 50))
        self.toolBar.setMovable(True)
        self.toolBar.setIconSize(QSize(30, 30))
        self.toolBar.setFloatable(False)
        mainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)
        self.toolBar_2 = QToolBar(mainWindow)
        self.toolBar_2.setObjectName(u"toolBar_2")
        self.toolBar_2.setLayoutDirection(Qt.RightToLeft)
        mainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar_2)

        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionClear_env)
        self.toolBar.addSeparator()
        self.toolBar_2.addAction(self.actionAbout)
        self.toolBar_2.addSeparator()
        self.toolBar_2.addAction(self.actionHelps)
        self.toolBar_2.addSeparator()
        self.toolBar_2.addAction(self.actionDarkMode)
        self.toolBar_2.addAction(self.actionLightMode)
        self.toolBar_2.addSeparator()

        self.retranslateUi(mainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_3.setCurrentIndex(0)
        self.cbb_wafer_size.setCurrentIndex(-1)
        self.tabWidget_2.setCurrentIndex(0)
        self.tabWidget_4.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(mainWindow)
    # setupUi

    def retranslateUi(self, mainWindow):
        mainWindow.setWindowTitle(QCoreApplication.translate("mainWindow", u"SPECTROview (Spectroscopic Data Processing and Visualization)", None))
        self.actionabout.setText(QCoreApplication.translate("mainWindow", u"About", None))
        self.actionOpen_dataframe_Excel.setText(QCoreApplication.translate("mainWindow", u"Open data inline (Semilab)", None))
        self.actionOpen_dataframe_CSV.setText(QCoreApplication.translate("mainWindow", u"Open data inline (Labspec6)", None))
        self.actionOpen_saved_work_s.setText(QCoreApplication.translate("mainWindow", u"Open saved work(s)", None))
        self.actionOpen_a_recipie.setText(QCoreApplication.translate("mainWindow", u"Open a recipie", None))
        self.actionSave_all_graph_PNG.setText(QCoreApplication.translate("mainWindow", u"Save all graphs (PNG)", None))
        self.actionSave_all_graphs_to_pptx.setText(QCoreApplication.translate("mainWindow", u"Save all graphs to pptx", None))
        self.open_df.setText(QCoreApplication.translate("mainWindow", u"Open df", None))
        self.actionHelps.setText(QCoreApplication.translate("mainWindow", u"Manual", None))
        self.actionDarkMode.setText(QCoreApplication.translate("mainWindow", u"Dark Mode", None))
        self.actionLightMode.setText(QCoreApplication.translate("mainWindow", u"Light Mode", None))
        self.actionAbout.setText(QCoreApplication.translate("mainWindow", u"About", None))
        self.actionOpen_wafer.setText(QCoreApplication.translate("mainWindow", u"Open hyperspectra : wafer, 2Dmap (CSV, txt)", None))
        self.action_reload.setText(QCoreApplication.translate("mainWindow", u"Reload saved work", None))
        self.actionOpen_spectra.setText(QCoreApplication.translate("mainWindow", u"Open spectra data (txt)", None))
        self.actionOpen_dfs.setText(QCoreApplication.translate("mainWindow", u"Open dataframe (Excel)", None))
        self.actionOpen.setText(QCoreApplication.translate("mainWindow", u"Open", None))
#if QT_CONFIG(tooltip)
        self.actionOpen.setToolTip(QCoreApplication.translate("mainWindow", u"Open spectra data, saved work or Excel file", None))
#endif // QT_CONFIG(tooltip)
        self.actionOpen_2.setText(QCoreApplication.translate("mainWindow", u"Open", None))
        self.actionSave.setText(QCoreApplication.translate("mainWindow", u"Save", None))
#if QT_CONFIG(tooltip)
        self.actionSave.setToolTip(QCoreApplication.translate("mainWindow", u"Save current work", None))
#endif // QT_CONFIG(tooltip)
        self.actionClear_WS.setText(QCoreApplication.translate("mainWindow", u"Clear WS", None))
        self.actionThem.setText(QCoreApplication.translate("mainWindow", u"Theme", None))
        self.actionClear_env.setText(QCoreApplication.translate("mainWindow", u"Clear env", None))
        self.actionLogo.setText(QCoreApplication.translate("mainWindow", u"zer", None))
        self.rdbtn_baseline_2.setText(QCoreApplication.translate("mainWindow", u"baseline", None))
        self.rdbtn_peak_2.setText(QCoreApplication.translate("mainWindow", u"peaks", None))
        self.rsquared_2.setText(QCoreApplication.translate("mainWindow", u"R2", None))
#if QT_CONFIG(tooltip)
        self.btn_copy_fig_3.setToolTip(QCoreApplication.translate("mainWindow", u"Copy Figure to clipboard", None))
#endif // QT_CONFIG(tooltip)
        self.btn_copy_fig_3.setText("")
        self.label_79.setText(QCoreApplication.translate("mainWindow", u"DPI:", None))
        self.toolButton.setText(QCoreApplication.translate("mainWindow", u"...", None))
        self.view_options_box_2.setTitle(QCoreApplication.translate("mainWindow", u"View options:", None))
        self.cb_residual_3.setText(QCoreApplication.translate("mainWindow", u"residual", None))
        self.cb_filled_3.setText(QCoreApplication.translate("mainWindow", u"filled", None))
        self.cb_bestfit_3.setText(QCoreApplication.translate("mainWindow", u"best-fit", None))
#if QT_CONFIG(tooltip)
        self.cb_legend_3.setToolTip(QCoreApplication.translate("mainWindow", u"To display or remove the legend of the plot", None))
#endif // QT_CONFIG(tooltip)
        self.cb_legend_3.setText(QCoreApplication.translate("mainWindow", u"legend", None))
        self.cb_raw_3.setText(QCoreApplication.translate("mainWindow", u"raw", None))
        self.cb_colors_3.setText(QCoreApplication.translate("mainWindow", u"colors", None))
        self.cb_peaks_3.setText(QCoreApplication.translate("mainWindow", u"peaks", None))
        self.cb_normalize_3.setText(QCoreApplication.translate("mainWindow", u"normalized", None))
#if QT_CONFIG(tooltip)
        self.btn_cosmis_ray_3.setToolTip(QCoreApplication.translate("mainWindow", u"Detect cosmis ray based on the loaded spectra", None))
#endif // QT_CONFIG(tooltip)
        self.btn_cosmis_ray_3.setText(QCoreApplication.translate("mainWindow", u"Spike removal", None))
        self.label_22.setText(QCoreApplication.translate("mainWindow", u"X-axis unit:", None))
        self.groupBox_5.setTitle("")
        self.label_65.setText(QCoreApplication.translate("mainWindow", u"Spectral range:", None))
        self.label_66.setText(QCoreApplication.translate("mainWindow", u"X min/max:", None))
        self.label_67.setText(QCoreApplication.translate("mainWindow", u"/", None))
#if QT_CONFIG(tooltip)
        self.range_apply_2.setToolTip(QCoreApplication.translate("mainWindow", u"Extract the spectral windows range.\n"
" Hold 'Ctrl' and press 'Extract' button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.range_apply_2.setText(QCoreApplication.translate("mainWindow", u"Extract", None))
        self.label_68.setText("")
        self.baseline_2.setTitle("")
        self.label_69.setText(QCoreApplication.translate("mainWindow", u"Baseline:", None))
        self.rbtn_linear_2.setText(QCoreApplication.translate("mainWindow", u"Linear", None))
        self.rbtn_polynomial_2.setText(QCoreApplication.translate("mainWindow", u"Poly", None))
        self.cb_attached_2.setText(QCoreApplication.translate("mainWindow", u"Attached", None))
#if QT_CONFIG(tooltip)
        self.label_70.setToolTip(QCoreApplication.translate("mainWindow", u"Number of nearby points considered to smoothing the noise", None))
#endif // QT_CONFIG(tooltip)
        self.label_70.setText(QCoreApplication.translate("mainWindow", u"Correct noise", None))
#if QT_CONFIG(tooltip)
        self.btn_undo_baseline_2.setToolTip(QCoreApplication.translate("mainWindow", u"To remove all baseline points and undo the baseline subtraction", None))
#endif // QT_CONFIG(tooltip)
        self.btn_undo_baseline_2.setText("")
#if QT_CONFIG(tooltip)
        self.btn_copy_bl_2.setToolTip(QCoreApplication.translate("mainWindow", u"Copy the baseline of the selected spectrum.  \n"
" Hold 'Ctrl' and press button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.btn_copy_bl_2.setText("")
#if QT_CONFIG(tooltip)
        self.btn_paste_bl_2.setToolTip(QCoreApplication.translate("mainWindow", u"Paste the copied baseline to the selected spectrum(s).  \n"
" Hold 'Ctrl' and press button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.btn_paste_bl_2.setText("")
#if QT_CONFIG(tooltip)
        self.sub_baseline_2.setToolTip(QCoreApplication.translate("mainWindow", u"Subtract baseline for selected spectrum(s). \n"
" Hold 'Ctrl' and press button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.sub_baseline_2.setText(QCoreApplication.translate("mainWindow", u"Subtract", None))
        self.label_71.setText("")
        self.peaks_2.setTitle("")
        self.label_72.setText(QCoreApplication.translate("mainWindow", u"Peaks:", None))
        self.label_73.setText(QCoreApplication.translate("mainWindow", u"Peak model:", None))
#if QT_CONFIG(tooltip)
        self.clear_peaks_2.setToolTip(QCoreApplication.translate("mainWindow", u"Clear all the current peak models. \n"
" Hold 'Ctrl' and press 'Clear peaks' button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.clear_peaks_2.setText(QCoreApplication.translate("mainWindow", u"Clear", None))
        self.peak_table_2.setTitle(QCoreApplication.translate("mainWindow", u"Peak table: ", None))
#if QT_CONFIG(tooltip)
        self.btn_fit_3.setToolTip(QCoreApplication.translate("mainWindow", u"Fit selected spectrum(s) with all parameters (range, baseline, peaks). \n"
" Hold 'Ctrl' and press 'Fit' button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.btn_fit_3.setText(QCoreApplication.translate("mainWindow", u"Fit", None))
#if QT_CONFIG(tooltip)
        self.btn_copy_fit_model_2.setToolTip(QCoreApplication.translate("mainWindow", u"Copy current fit parameters (range, baseline, peaks) of selected spectrum to Clipboard", None))
#endif // QT_CONFIG(tooltip)
        self.btn_copy_fit_model_2.setText(QCoreApplication.translate("mainWindow", u"Copy model", None))
        self.lbl_copied_fit_model_2.setText("")
#if QT_CONFIG(tooltip)
        self.btn_paste_fit_model_2.setToolTip(QCoreApplication.translate("mainWindow", u"Paste the copied fit model to the selected spectrum(s).\n"
" Hold 'Ctrl' and press 'Paste' button to paste fit model to all spectra, including spectra within different wafers", None))
#endif // QT_CONFIG(tooltip)
        self.btn_paste_fit_model_2.setText(QCoreApplication.translate("mainWindow", u"Paste model", None))
#if QT_CONFIG(tooltip)
        self.save_model_2.setToolTip(QCoreApplication.translate("mainWindow", u"Save the fit model as a JSON file", None))
#endif // QT_CONFIG(tooltip)
        self.save_model_2.setText(QCoreApplication.translate("mainWindow", u"Save Model", None))
#if QT_CONFIG(tooltip)
        self.cb_limits_2.setToolTip(QCoreApplication.translate("mainWindow", u"Show limits (max, min) of each parameters", None))
#endif // QT_CONFIG(tooltip)
        self.cb_limits_2.setText(QCoreApplication.translate("mainWindow", u"Limits", None))
#if QT_CONFIG(tooltip)
        self.cb_expr_2.setToolTip(QCoreApplication.translate("mainWindow", u"Show the expression of fit parameters", None))
#endif // QT_CONFIG(tooltip)
        self.cb_expr_2.setText(QCoreApplication.translate("mainWindow", u"Expression", None))
        self.label_81.setText(QCoreApplication.translate("mainWindow", u"Select a model:", None))
        self.cbb_fit_model_list_3.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Select a model for fitting", None))
        self.btn_apply_model_3.setText(QCoreApplication.translate("mainWindow", u"Apply model", None))
#if QT_CONFIG(tooltip)
        self.btn_load_model_3.setToolTip(QCoreApplication.translate("mainWindow", u"Load a fit model if it is not in the default folder", None))
#endif // QT_CONFIG(tooltip)
        self.btn_load_model_3.setText(QCoreApplication.translate("mainWindow", u"Load model", None))
        self.tabWidget_3.setTabText(self.tabWidget_3.indexOf(self.fit_model_editor_3), QCoreApplication.translate("mainWindow", u"Fit model builder", None))
#if QT_CONFIG(tooltip)
        self.btn_collect_results_3.setToolTip(QCoreApplication.translate("mainWindow", u"To gather all the best fit results in a dataframe for visualization", None))
#endif // QT_CONFIG(tooltip)
        self.btn_collect_results_3.setText(QCoreApplication.translate("mainWindow", u" Collect fit results", None))
        self.label_83.setText(QCoreApplication.translate("mainWindow", u"Add new column(s) from file name:", None))
#if QT_CONFIG(tooltip)
        self.btn_split_fname.setToolTip(QCoreApplication.translate("mainWindow", u"Split the file name of spectrum to several parts", None))
#endif // QT_CONFIG(tooltip)
        self.btn_split_fname.setText(QCoreApplication.translate("mainWindow", u"Split", None))
#if QT_CONFIG(tooltip)
        self.cbb_split_fname.setToolTip(QCoreApplication.translate("mainWindow", u"Select file name part", None))
#endif // QT_CONFIG(tooltip)
        self.cbb_split_fname.setPlaceholderText("")
        self.ent_col_name.setPlaceholderText(QCoreApplication.translate("mainWindow", u"type column name", None))
#if QT_CONFIG(tooltip)
        self.btn_add_col.setToolTip(QCoreApplication.translate("mainWindow", u"Add a new column containing the selected part", None))
#endif // QT_CONFIG(tooltip)
        self.btn_add_col.setText(QCoreApplication.translate("mainWindow", u"Add", None))
        self.ent_send_df_to_viz.setText(QCoreApplication.translate("mainWindow", u"SPECTRUMS_best_fit", None))
        self.btn_send_to_viz.setText(QCoreApplication.translate("mainWindow", u"Send to Visualization", None))
#if QT_CONFIG(tooltip)
        self.btn_view_df_5.setToolTip(QCoreApplication.translate("mainWindow", u"View collected fit results", None))
#endif // QT_CONFIG(tooltip)
        self.btn_view_df_5.setText("")
#if QT_CONFIG(tooltip)
        self.btn_save_fit_results_3.setToolTip(QCoreApplication.translate("mainWindow", u"Save all fit results in an Excel file", None))
#endif // QT_CONFIG(tooltip)
        self.btn_save_fit_results_3.setText("")
#if QT_CONFIG(tooltip)
        self.btn_open_fit_results_3.setToolTip(QCoreApplication.translate("mainWindow", u"Open fit results (format Excel) to view/plot", None))
#endif // QT_CONFIG(tooltip)
        self.btn_open_fit_results_3.setText("")
        self.groupBox_6.setTitle("")
        self.tabWidget_3.setTabText(self.tabWidget_3.indexOf(self.collect_fit_data_2), QCoreApplication.translate("mainWindow", u"Collect fit data", None))
        self.label_74.setText(QCoreApplication.translate("mainWindow", u"Fit settings:", None))
        self.label_51.setText(QCoreApplication.translate("mainWindow", u"Fit negative values:", None))
        self.cb_fit_negative_2.setText("")
        self.label_75.setText(QCoreApplication.translate("mainWindow", u"Maximumn iterations :", None))
        self.max_iteration_2.setPrefix("")
        self.label_76.setText(QCoreApplication.translate("mainWindow", u"Fit method", None))
        self.label_78.setText(QCoreApplication.translate("mainWindow", u"x-tolerence", None))
        self.xtol_2.setText(QCoreApplication.translate("mainWindow", u"0.0001", None))
#if QT_CONFIG(tooltip)
        self.btn_open_fitspy_3.setToolTip(QCoreApplication.translate("mainWindow", u"Open FITSPY application", None))
#endif // QT_CONFIG(tooltip)
        self.btn_open_fitspy_3.setText(QCoreApplication.translate("mainWindow", u"Open FITSPY", None))
        self.btn_default_folder_model_3.setText(QCoreApplication.translate("mainWindow", u"Model Folder:", None))
        self.btn_refresh_model_folder_3.setText(QCoreApplication.translate("mainWindow", u"refresh", None))
        self.tabWidget_3.setTabText(self.tabWidget_3.indexOf(self.fit_settings_3), QCoreApplication.translate("mainWindow", u"More settings", None))
        self.groupBox_12.setTitle(QCoreApplication.translate("mainWindow", u"Spectrum(s):", None))
#if QT_CONFIG(tooltip)
        self.btn_sel_all_3.setToolTip(QCoreApplication.translate("mainWindow", u"Select all spectre", None))
#endif // QT_CONFIG(tooltip)
        self.btn_sel_all_3.setText("")
        self.btn_remove_spectrum.setText("")
#if QT_CONFIG(tooltip)
        self.btn_init_3.setToolTip(QCoreApplication.translate("mainWindow", u"Reinitialize to RAW spectrum (Hold Ctrl to reinit all spectra)", None))
#endif // QT_CONFIG(tooltip)
        self.btn_init_3.setText(QCoreApplication.translate("mainWindow", u"Reinitialize", None))
#if QT_CONFIG(tooltip)
        self.btn_show_stats_3.setToolTip(QCoreApplication.translate("mainWindow", u"Show fitting statistique results", None))
#endif // QT_CONFIG(tooltip)
        self.btn_show_stats_3.setText(QCoreApplication.translate("mainWindow", u"Stats", None))
        self.checkBox.setText(QCoreApplication.translate("mainWindow", u"Check/Uncheck all", None))
        self.item_count_label_3.setText(QCoreApplication.translate("mainWindow", u"0 points", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_spectra), QCoreApplication.translate("mainWindow", u"Spectra", None))
        self.rdbtn_baseline.setText(QCoreApplication.translate("mainWindow", u"baseline", None))
        self.rdbtn_peak.setText(QCoreApplication.translate("mainWindow", u"peaks", None))
        self.rsquared_1.setText(QCoreApplication.translate("mainWindow", u"R2=0 ", None))
#if QT_CONFIG(tooltip)
        self.btn_copy_fig.setToolTip(QCoreApplication.translate("mainWindow", u"Copy Figure to clipboard", None))
#endif // QT_CONFIG(tooltip)
        self.btn_copy_fig.setText("")
        self.label_63.setText(QCoreApplication.translate("mainWindow", u"DPI:", None))
        self.rdbt_show_2Dmap.setText(QCoreApplication.translate("mainWindow", u"2D map", None))
        self.rdbt_show_wafer.setText(QCoreApplication.translate("mainWindow", u"Wafer", None))
#if QT_CONFIG(tooltip)
        self.cbb_wafer_size.setToolTip(QCoreApplication.translate("mainWindow", u"Wafer diamters (nm)", None))
#endif // QT_CONFIG(tooltip)
        self.cbb_wafer_size.setCurrentText("")
        self.cbb_wafer_size.setPlaceholderText("")
        self.view_options_box.setTitle(QCoreApplication.translate("mainWindow", u"View options:", None))
        self.cb_residual.setText(QCoreApplication.translate("mainWindow", u"residual", None))
        self.cb_filled.setText(QCoreApplication.translate("mainWindow", u"filled", None))
        self.cb_bestfit.setText(QCoreApplication.translate("mainWindow", u"best-fit", None))
#if QT_CONFIG(tooltip)
        self.cb_legend.setToolTip(QCoreApplication.translate("mainWindow", u"To display or remove the legend of the plot", None))
#endif // QT_CONFIG(tooltip)
        self.cb_legend.setText(QCoreApplication.translate("mainWindow", u"legend", None))
        self.cb_raw.setText(QCoreApplication.translate("mainWindow", u"raw", None))
        self.cb_colors.setText(QCoreApplication.translate("mainWindow", u"colors", None))
        self.cb_peaks.setText(QCoreApplication.translate("mainWindow", u"peaks", None))
        self.cb_normalize.setText(QCoreApplication.translate("mainWindow", u"normalized", None))
#if QT_CONFIG(tooltip)
        self.btn_cosmis_ray.setToolTip(QCoreApplication.translate("mainWindow", u"Detect cosmis ray based on the loaded spectra", None))
#endif // QT_CONFIG(tooltip)
        self.btn_cosmis_ray.setText(QCoreApplication.translate("mainWindow", u"Spike removal", None))
        self.label_99.setText(QCoreApplication.translate("mainWindow", u"X-axis unit:", None))
        self.groupBox_4.setTitle("")
        self.label_54.setText(QCoreApplication.translate("mainWindow", u"Spectral range:", None))
        self.label_61.setText(QCoreApplication.translate("mainWindow", u"X min/max:", None))
        self.label_62.setText(QCoreApplication.translate("mainWindow", u"/", None))
#if QT_CONFIG(tooltip)
        self.range_apply.setToolTip(QCoreApplication.translate("mainWindow", u"Extract the spectral windows range.\n"
" Hold 'Ctrl' and press 'Extract' button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.range_apply.setText(QCoreApplication.translate("mainWindow", u"Extract", None))
        self.label_59.setText("")
        self.baseline.setTitle("")
        self.label_52.setText(QCoreApplication.translate("mainWindow", u"Baseline:", None))
        self.rbtn_linear.setText(QCoreApplication.translate("mainWindow", u"Linear", None))
        self.rbtn_polynomial.setText(QCoreApplication.translate("mainWindow", u"Poly", None))
        self.cb_attached.setText(QCoreApplication.translate("mainWindow", u"Attached", None))
#if QT_CONFIG(tooltip)
        self.label_37.setToolTip(QCoreApplication.translate("mainWindow", u"Number of nearby points considered to smoothing the noise", None))
#endif // QT_CONFIG(tooltip)
        self.label_37.setText(QCoreApplication.translate("mainWindow", u"Correct Noise", None))
#if QT_CONFIG(tooltip)
        self.btn_undo_baseline.setToolTip(QCoreApplication.translate("mainWindow", u"To remove all baseline points and undo the baseline subtraction", None))
#endif // QT_CONFIG(tooltip)
        self.btn_undo_baseline.setText("")
#if QT_CONFIG(tooltip)
        self.btn_copy_bl.setToolTip(QCoreApplication.translate("mainWindow", u"Copy the baseline of the selected spectrum.  \n"
" Hold 'Ctrl' and press button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.btn_copy_bl.setText("")
#if QT_CONFIG(tooltip)
        self.btn_paste_bl.setToolTip(QCoreApplication.translate("mainWindow", u"Paste the copied baseline to the selected spectrum(s).  \n"
" Hold 'Ctrl' and press button to apply to all spectra.", None))
#endif // QT_CONFIG(tooltip)
        self.btn_paste_bl.setText("")
#if QT_CONFIG(tooltip)
        self.sub_baseline.setToolTip(QCoreApplication.translate("mainWindow", u"Subtract baseline for selected spectrum(s). \n"
" Hold 'Ctrl' and press button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.sub_baseline.setText(QCoreApplication.translate("mainWindow", u"Subtract", None))
        self.label_60.setText("")
        self.peaks.setTitle("")
        self.label_57.setText(QCoreApplication.translate("mainWindow", u"Peaks:", None))
        self.label_41.setText(QCoreApplication.translate("mainWindow", u"Peak model:", None))
#if QT_CONFIG(tooltip)
        self.clear_peaks.setToolTip(QCoreApplication.translate("mainWindow", u"Clear all the current peak models. \n"
" Hold 'Ctrl' and press 'Clear peaks' button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.clear_peaks.setText("")
        self.peak_table.setTitle(QCoreApplication.translate("mainWindow", u"Peak table: ", None))
#if QT_CONFIG(tooltip)
        self.btn_fit.setToolTip(QCoreApplication.translate("mainWindow", u"Fit selected spectrum(s) with all parameters (range, baseline, peaks). \n"
" Hold 'Ctrl' and press 'Fit' button to apply to all spectra", None))
#endif // QT_CONFIG(tooltip)
        self.btn_fit.setText(QCoreApplication.translate("mainWindow", u"Fit", None))
#if QT_CONFIG(tooltip)
        self.btn_copy_fit_model.setToolTip(QCoreApplication.translate("mainWindow", u"Copy current fit parameters (range, baseline, peaks) of selected spectrum to Clipboard", None))
#endif // QT_CONFIG(tooltip)
        self.btn_copy_fit_model.setText(QCoreApplication.translate("mainWindow", u"Copy model", None))
        self.lbl_copied_fit_model.setText("")
#if QT_CONFIG(tooltip)
        self.btn_paste_fit_model.setToolTip(QCoreApplication.translate("mainWindow", u"Paste the copied fit model to the selected spectrum(s).\n"
" Hold 'Ctrl' and press 'Paste' button to paste fit model to all spectra, including spectra within different wafers", None))
#endif // QT_CONFIG(tooltip)
        self.btn_paste_fit_model.setText(QCoreApplication.translate("mainWindow", u"Paste model", None))
#if QT_CONFIG(tooltip)
        self.save_model.setToolTip(QCoreApplication.translate("mainWindow", u"Save the fit model as a JSON file", None))
#endif // QT_CONFIG(tooltip)
        self.save_model.setText(QCoreApplication.translate("mainWindow", u"Save Model", None))
#if QT_CONFIG(tooltip)
        self.cb_limits.setToolTip(QCoreApplication.translate("mainWindow", u"Show limits (max, min) of each parameters", None))
#endif // QT_CONFIG(tooltip)
        self.cb_limits.setText(QCoreApplication.translate("mainWindow", u"Limits", None))
#if QT_CONFIG(tooltip)
        self.cb_expr.setToolTip(QCoreApplication.translate("mainWindow", u"Show the expression of fit parameters", None))
#endif // QT_CONFIG(tooltip)
        self.cb_expr.setText(QCoreApplication.translate("mainWindow", u"Expression", None))
        self.label_80.setText(QCoreApplication.translate("mainWindow", u"Select a model:", None))
#if QT_CONFIG(tooltip)
        self.cbb_fit_model_list.setToolTip(QCoreApplication.translate("mainWindow", u"Go to 'Settings' to specify a folder where fit models are stored ", None))
#endif // QT_CONFIG(tooltip)
        self.cbb_fit_model_list.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Select a model for fitting", None))
#if QT_CONFIG(tooltip)
        self.btn_apply_model.setToolTip(QCoreApplication.translate("mainWindow", u"Hold \"Ctrl\" and click to apply fit model to all wafer", None))
#endif // QT_CONFIG(tooltip)
        self.btn_apply_model.setText(QCoreApplication.translate("mainWindow", u"Apply model", None))
#if QT_CONFIG(tooltip)
        self.btn_load_model.setToolTip(QCoreApplication.translate("mainWindow", u"Load a fit model if it is not in the default folder", None))
#endif // QT_CONFIG(tooltip)
        self.btn_load_model.setText(QCoreApplication.translate("mainWindow", u"Load model", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.fit_model_editor), QCoreApplication.translate("mainWindow", u"Fit model builder", None))
#if QT_CONFIG(tooltip)
        self.btn_collect_results.setToolTip(QCoreApplication.translate("mainWindow", u"To gather all the best fit results in a dataframe for visualization", None))
#endif // QT_CONFIG(tooltip)
        self.btn_collect_results.setText(QCoreApplication.translate("mainWindow", u" Collect fit results", None))
        self.label_56.setText(QCoreApplication.translate("mainWindow", u"Add new column(s) from file name:", None))
#if QT_CONFIG(tooltip)
        self.btn_split_fname_2.setToolTip(QCoreApplication.translate("mainWindow", u"Split the file name of spectrum to several parts", None))
#endif // QT_CONFIG(tooltip)
        self.btn_split_fname_2.setText(QCoreApplication.translate("mainWindow", u"Split", None))
#if QT_CONFIG(tooltip)
        self.cbb_split_fname_2.setToolTip(QCoreApplication.translate("mainWindow", u"Select file name part", None))
#endif // QT_CONFIG(tooltip)
        self.cbb_split_fname_2.setPlaceholderText("")
        self.ent_col_name_2.setPlaceholderText(QCoreApplication.translate("mainWindow", u"type column name", None))
#if QT_CONFIG(tooltip)
        self.btn_add_col_2.setToolTip(QCoreApplication.translate("mainWindow", u"Add a new column containing the selected part", None))
#endif // QT_CONFIG(tooltip)
        self.btn_add_col_2.setText(QCoreApplication.translate("mainWindow", u"Add", None))
        self.ent_send_df_to_viz2.setText(QCoreApplication.translate("mainWindow", u"MAPS_best_fit", None))
        self.btn_send_to_viz2.setText(QCoreApplication.translate("mainWindow", u"Send to Visualization", None))
#if QT_CONFIG(tooltip)
        self.btn_view_df_2.setToolTip(QCoreApplication.translate("mainWindow", u"View collected fit results", None))
#endif // QT_CONFIG(tooltip)
        self.btn_view_df_2.setText("")
#if QT_CONFIG(tooltip)
        self.btn_save_fit_results.setToolTip(QCoreApplication.translate("mainWindow", u"Save all fit results in an Excel file", None))
#endif // QT_CONFIG(tooltip)
        self.btn_save_fit_results.setText("")
#if QT_CONFIG(tooltip)
        self.btn_open_fit_results.setToolTip(QCoreApplication.translate("mainWindow", u"Open fit results (format Excel) to view/plot", None))
#endif // QT_CONFIG(tooltip)
        self.btn_open_fit_results.setText("")
        self.groupBox_3.setTitle("")
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.collect_fit_data), QCoreApplication.translate("mainWindow", u"Collect fit data", None))
        self.groupBox_8.setTitle(QCoreApplication.translate("mainWindow", u"Peak position correction :", None))
        self.radioButton_3.setText("")
        self.label_45.setText(QCoreApplication.translate("mainWindow", u"No", None))
        self.radioButton_2.setText("")
        self.label_44.setText(QCoreApplication.translate("mainWindow", u"Peak  shift (cm-1) :", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_32.setToolTip(QCoreApplication.translate("mainWindow", u"Type the different between experimental and theoretical values", None))
#endif // QT_CONFIG(tooltip)
        self.label_47.setText(QCoreApplication.translate("mainWindow", u"Reference peak :", None))
#if QT_CONFIG(tooltip)
        self.lineEdit_34.setToolTip(QCoreApplication.translate("mainWindow", u"Type the theoretical values", None))
#endif // QT_CONFIG(tooltip)
        self.label_23.setText(QCoreApplication.translate("mainWindow", u"Fit negative values:", None))
        self.cb_fit_negative.setText("")
        self.label_25.setText(QCoreApplication.translate("mainWindow", u"Maximumn iterations :", None))
        self.max_iteration.setPrefix("")
        self.label_27.setText(QCoreApplication.translate("mainWindow", u"Fit method", None))
        self.label_55.setText(QCoreApplication.translate("mainWindow", u"x-tolerence", None))
        self.xtol.setText(QCoreApplication.translate("mainWindow", u"0.0001", None))
        self.btn_open_fitspy.setText(QCoreApplication.translate("mainWindow", u"open FITSPY", None))
        self.label_53.setText(QCoreApplication.translate("mainWindow", u"Fit settings:", None))
        self.btn_default_folder_model.setText(QCoreApplication.translate("mainWindow", u"Model Folder", None))
        self.btn_refresh_model_folder.setText(QCoreApplication.translate("mainWindow", u"refresh", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.fit_settings), QCoreApplication.translate("mainWindow", u"More settings", None))
        self.groupBox.setTitle("")
        self.label_64.setText(QCoreApplication.translate("mainWindow", u"Maps:", None))
        self.btn_remove_wafer.setText("")
        self.btn_view_wafer.setText("")
        self.pushButton_3.setText("")
        self.groupBox_2.setTitle("")
        self.label_58.setText(QCoreApplication.translate("mainWindow", u"Quick selection :", None))
#if QT_CONFIG(tooltip)
        self.btn_sel_all.setToolTip(QCoreApplication.translate("mainWindow", u"Select all spectre", None))
#endif // QT_CONFIG(tooltip)
        self.btn_sel_all.setText("")
#if QT_CONFIG(tooltip)
        self.btn_sel_verti.setToolTip(QCoreApplication.translate("mainWindow", u"Select vertical", None))
#endif // QT_CONFIG(tooltip)
        self.btn_sel_verti.setText("")
#if QT_CONFIG(tooltip)
        self.btn_sel_horiz.setToolTip(QCoreApplication.translate("mainWindow", u"Select horizontal", None))
#endif // QT_CONFIG(tooltip)
        self.btn_sel_horiz.setText("")
        self.btn_sel_q1.setText(QCoreApplication.translate("mainWindow", u"Q1", None))
        self.btn_sel_q2.setText(QCoreApplication.translate("mainWindow", u"Q2", None))
        self.btn_sel_q3.setText(QCoreApplication.translate("mainWindow", u"Q3", None))
        self.btn_sel_q4.setText(QCoreApplication.translate("mainWindow", u"Q4", None))
        self.checkBox_2.setText(QCoreApplication.translate("mainWindow", u"Check/Uncheck all", None))
#if QT_CONFIG(tooltip)
        self.btn_init.setToolTip(QCoreApplication.translate("mainWindow", u"Reinitialize to RAW spectrum (Hold Ctrl to reinit all spectra)", None))
#endif // QT_CONFIG(tooltip)
        self.btn_init.setText(QCoreApplication.translate("mainWindow", u"Reinitialize", None))
#if QT_CONFIG(tooltip)
        self.btn_show_stats.setToolTip(QCoreApplication.translate("mainWindow", u"Show fitting statistique results", None))
#endif // QT_CONFIG(tooltip)
        self.btn_show_stats.setText(QCoreApplication.translate("mainWindow", u"Stats", None))
        self.btn_send_to_compare.setText(QCoreApplication.translate("mainWindow", u"Send selected spectrum(s) to 'Spectra' TAB", None))
        self.item_count_label.setText(QCoreApplication.translate("mainWindow", u"0 points", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_maps), QCoreApplication.translate("mainWindow", u"Maps", None))
        self.cbb_graph_list.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Graph list", None))
        self.btn_minimize_all.setText(QCoreApplication.translate("mainWindow", u"Minimize All", None))
        self.lbl_figsize.setText("")
        self.label_16.setText(QCoreApplication.translate("mainWindow", u"DPI:", None))
        self.label_92.setText(QCoreApplication.translate("mainWindow", u"X label rotation (\u00b0):", None))
        self.cb_legend_outside.setText(QCoreApplication.translate("mainWindow", u"Legend outside", None))
        self.cb_grid.setText(QCoreApplication.translate("mainWindow", u"Grid", None))
        self.btn_copy_graph.setText("")
        self.groupBox_loaded_df_2.setTitle(QCoreApplication.translate("mainWindow", u"Loaded dataframe(s):", None))
        self.merge_dfs_2.setText(QCoreApplication.translate("mainWindow", u"Merge all", None))
#if QT_CONFIG(tooltip)
        self.btn_view_df_3.setToolTip(QCoreApplication.translate("mainWindow", u"View selected dataframe \n"
"(Hold CTRL key to view filtered df)", None))
#endif // QT_CONFIG(tooltip)
        self.btn_view_df_3.setText("")
#if QT_CONFIG(tooltip)
        self.btn_remove_df_2.setToolTip(QCoreApplication.translate("mainWindow", u"Remove selected dataframe", None))
#endif // QT_CONFIG(tooltip)
        self.btn_remove_df_2.setText("")
#if QT_CONFIG(tooltip)
        self.btn_save_df_2.setToolTip(QCoreApplication.translate("mainWindow", u"Save all df to one Excel files \n"
"(Hold Ctrl to save df to seperated files)", None))
#endif // QT_CONFIG(tooltip)
        self.btn_save_df_2.setText("")
        self.groupBox_df_manip_5.setTitle(QCoreApplication.translate("mainWindow", u"Data filtering:", None))
        self.filter_query.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Enter query expression (? : see Help menu)", None))
#if QT_CONFIG(tooltip)
        self.btn_add_filter_4.setToolTip(QCoreApplication.translate("mainWindow", u"Add filter(s) to dataframe", None))
#endif // QT_CONFIG(tooltip)
        self.btn_add_filter_4.setText("")
#if QT_CONFIG(tooltip)
        self.btn_remove_filters_4.setToolTip(QCoreApplication.translate("mainWindow", u"Remove filters from the list", None))
#endif // QT_CONFIG(tooltip)
        self.btn_remove_filters_4.setText("")
#if QT_CONFIG(tooltip)
        self.btn_apply_filters_4.setToolTip(QCoreApplication.translate("mainWindow", u"Apply selected filters to selected dataframe", None))
#endif // QT_CONFIG(tooltip)
        self.btn_apply_filters_4.setText(QCoreApplication.translate("mainWindow", u"Apply", None))
        self.btn_add_graph.setText(QCoreApplication.translate("mainWindow", u"Add plot", None))
        self.btn_upd_graph.setText(QCoreApplication.translate("mainWindow", u" Update plot", None))
        self.btn_add_line.setText(QCoreApplication.translate("mainWindow", u"RECIPE", None))
        self.label_96.setText(QCoreApplication.translate("mainWindow", u"Plot styles:", None))
        self.label_93.setText(QCoreApplication.translate("mainWindow", u"Color palette:", None))
        self.label_82.setText(QCoreApplication.translate("mainWindow", u"X", None))
        self.label_84.setText(QCoreApplication.translate("mainWindow", u"Y", None))
        self.label_85.setText(QCoreApplication.translate("mainWindow", u"Z", None))
        self.label.setText("")
        self.label_91.setText(QCoreApplication.translate("mainWindow", u"Plot title: ", None))
        self.lbl_plot_title.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Type to modify the plot title", None))
        self.label_86.setText(QCoreApplication.translate("mainWindow", u"X label", None))
        self.lbl_xlabel.setPlaceholderText(QCoreApplication.translate("mainWindow", u"X axis label", None))
        self.label_87.setText(QCoreApplication.translate("mainWindow", u"Y label", None))
        self.lbl_ylabel.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Y axis label", None))
        self.label_97.setText(QCoreApplication.translate("mainWindow", u"Axis limits:", None))
        self.btn_get_limits.setText(QCoreApplication.translate("mainWindow", u"Set current limits", None))
        self.btn_clear_limits.setText(QCoreApplication.translate("mainWindow", u"Clear limits", None))
        self.label_89.setText(QCoreApplication.translate("mainWindow", u"X limits", None))
        self.xmin_2.setPlaceholderText(QCoreApplication.translate("mainWindow", u"X min", None))
        self.xmax_2.setPlaceholderText(QCoreApplication.translate("mainWindow", u"X max", None))
        self.label_90.setText(QCoreApplication.translate("mainWindow", u"Y limits", None))
        self.ymin_2.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Y min", None))
        self.ymax_2.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Y max", None))
        self.label_2.setText("")
        self.label_88.setText(QCoreApplication.translate("mainWindow", u"Z label", None))
        self.lbl_zlabel.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Z axis label", None))
        self.label_94.setText(QCoreApplication.translate("mainWindow", u"Z limits / color range", None))
        self.zmin_2.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Z min", None))
        self.zmax_2.setPlaceholderText(QCoreApplication.translate("mainWindow", u"Z max", None))
        self.label_3.setText("")
        self.label_98.setText(QCoreApplication.translate("mainWindow", u"Wafer size (mm)", None))
        self.lbl_wafersize.setText(QCoreApplication.translate("mainWindow", u"300", None))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_plot_settings), QCoreApplication.translate("mainWindow", u"Plot Settings", None))
        self.cb_legend_visible.setText(QCoreApplication.translate("mainWindow", u"Show legend", None))
        self.cb_show_err_bar_plot.setText(QCoreApplication.translate("mainWindow", u"error bar for 'bar_plot'", None))
        self.cb_wafer_stats.setText(QCoreApplication.translate("mainWindow", u"stats on 'wafer_plot'", None))
        self.cb_join_for_point_plot.setText(QCoreApplication.translate("mainWindow", u"join for 'point_plot'", None))
        self.cb_trendline_eq.setText(QCoreApplication.translate("mainWindow", u"add trendline equation (oder", None))
        self.label_18.setText(QCoreApplication.translate("mainWindow", u")", None))
        self.label_17.setText(QCoreApplication.translate("mainWindow", u"Legend location: ", None))
        self.label_13.setText(QCoreApplication.translate("mainWindow", u"Customize color and legend:", None))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_more_options), QCoreApplication.translate("mainWindow", u"More options", None))
        self.label_12.setText(QCoreApplication.translate("mainWindow", u"Multiples Y-axis:", None))
        self.label100.setText(QCoreApplication.translate("mainWindow", u"Y2:", None))
        self.btn_add_y2.setText(QCoreApplication.translate("mainWindow", u"Add", None))
        self.btn_remove_y2.setText(QCoreApplication.translate("mainWindow", u"Remove", None))
        self.label_4.setText(QCoreApplication.translate("mainWindow", u"Y2 label:", None))
        self.label_5.setText(QCoreApplication.translate("mainWindow", u"Y2 min:", None))
        self.label_6.setText(QCoreApplication.translate("mainWindow", u"Y2 max:", None))
        self.label_11.setText(QCoreApplication.translate("mainWindow", u"Y3:", None))
        self.btn_add_y3.setText(QCoreApplication.translate("mainWindow", u"Add ", None))
        self.btn_remove_y3.setText(QCoreApplication.translate("mainWindow", u"Remove", None))
        self.label_7.setText(QCoreApplication.translate("mainWindow", u"Y3 label:", None))
        self.label_9.setText(QCoreApplication.translate("mainWindow", u"Y3 min:", None))
        self.label_8.setText(QCoreApplication.translate("mainWindow", u"Y3 max:", None))
        self.label100_2.setText(QCoreApplication.translate("mainWindow", u"Y12:", None))
        self.btn_add_y12.setText(QCoreApplication.translate("mainWindow", u"Add", None))
        self.btn_remove_y12.setText(QCoreApplication.translate("mainWindow", u"Remove", None))
        self.label_10.setText(QCoreApplication.translate("mainWindow", u"Y12 label:", None))
        self.label_14.setText(QCoreApplication.translate("mainWindow", u"Y13:", None))
        self.btn_add_y13.setText(QCoreApplication.translate("mainWindow", u"Add ", None))
        self.btn_remove_y13.setText(QCoreApplication.translate("mainWindow", u"Remove", None))
        self.label_15.setText(QCoreApplication.translate("mainWindow", u"Y13 label:", None))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_multi_axes), QCoreApplication.translate("mainWindow", u"Multi Yaxis/Lines (beta)", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_graphs), QCoreApplication.translate("mainWindow", u"Data Visualization", None))
        self.label_38.setText("")
        self.label_19.setText(QCoreApplication.translate("mainWindow", u"Number of CPUs:", None))
        self.label_21.setText("")
        self.progressText.setText("")
        self.label_95.setText("")
        self.progressBar.setFormat(QCoreApplication.translate("mainWindow", u"%p%", None))
        self.label_24.setText("")
        self.toolBar.setWindowTitle(QCoreApplication.translate("mainWindow", u"toolBar", None))
        self.toolBar_2.setWindowTitle(QCoreApplication.translate("mainWindow", u"toolBar_2", None))
    # retranslateUi

