import os
import markdown
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTextBrowser,
    QLineEdit, QWidget, QListWidget, QListWidgetItem, QLabel,
    QPushButton
)
from PySide6.QtGui import (
    QDesktopServices, QTextDocument, QImage,
    QTextCursor, QMovie
)

class FitImageTextBrowser(QTextBrowser):
    """A QTextBrowser that automatically scales down large images
    to fit the viewport width and supports animated GIF playback
    via QLabel overlays."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)
        self._orig_sizes = {}
        self._movies = {}             # name -> QMovie
        self._gif_labels = {}         # name -> QLabel overlay
        self._gif_doc_positions = {}  # name -> char position in doc
        self._scroll_connected = False
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_images()
        self._reposition_gif_labels()

    def _resize_images(self):
        doc = self.document()
        max_width = self.viewport().width() - 30
        if max_width <= 0: return
        
        # Save scroll positions to prevent jumping
        v_scroll = self.verticalScrollBar().value()
        h_scroll = self.horizontalScrollBar().value()
        
        cursor = QTextCursor(doc)
        cursor.setPosition(0)
        
        block = doc.begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                if fragment.isValid():
                    fmt = fragment.charFormat()
                    if fmt.isImageFormat():
                        img_fmt = fmt.toImageFormat()
                        name = img_fmt.name()
                        
                        if name not in self._orig_sizes:
                            url = QUrl(name)
                            if url.isRelative():
                                url = doc.baseUrl().resolved(url)
                            img = QImage(url.toLocalFile())
                            if not img.isNull():
                                self._orig_sizes[name] = img.size()
                                
                        orig_size = self._orig_sizes.get(name)
                        if orig_size:
                            new_width = min(orig_size.width(), max_width)
                            if img_fmt.width() != new_width:
                                img_fmt.setWidth(new_width)
                                aspect = orig_size.height() / orig_size.width()
                                new_height = new_width * aspect
                                img_fmt.setHeight(new_height)
                                
                                # For static images, add a smooth-scaled resource
                                if not name.lower().endswith('.gif'):
                                    url = QUrl(name)
                                    if url.isRelative():
                                        url = doc.baseUrl().resolved(url)
                                    img = QImage(url.toLocalFile())
                                    if not img.isNull():
                                        scaled_img = img.scaled(
                                            new_width, int(new_height),
                                            Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation
                                        )
                                        doc.addResource(QTextDocument.ResourceType.ImageResource, url, scaled_img)
                                
                                cursor.setPosition(fragment.position())
                                cursor.setPosition(fragment.position() + fragment.length(), QTextCursor.MoveMode.KeepAnchor)
                                cursor.setCharFormat(img_fmt)
                it += 1
            block = block.next()
        
        # Restore the scroll positions
        self.verticalScrollBar().setValue(v_scroll)
        self.horizontalScrollBar().setValue(h_scroll)

    # ------------------------------------------------------------------
    # Animated GIF support  (QLabel overlay approach)
    # ------------------------------------------------------------------
    def cleanup_movies(self):
        """Stop all movies and destroy overlay labels."""
        # Use list() to avoid issues with modification during iteration
        for name in list(self._gif_labels.keys()):
            label = self._gif_labels.pop(name)
            label.hide()
            label.setParent(None)
            label.deleteLater()
        
        for movie in list(self._movies.values()):
            movie.stop()
        
        self._movies.clear()
        self._gif_labels.clear()
        self._gif_doc_positions.clear()

    def setup_gif_animations(self):
        """Scan the rendered document for .gif images.  For each one,
        create a QLabel + QMovie overlay on top of the static
        placeholder so the animation plays natively via Qt."""
        doc = self.document()
        base_url = doc.baseUrl()

        block = doc.begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                if fragment.isValid():
                    fmt = fragment.charFormat()
                    if fmt.isImageFormat():
                        img_fmt = fmt.toImageFormat()
                        name = img_fmt.name()

                        if name.lower().endswith('.gif') and name not in self._movies:
                            url = QUrl(name)
                            if url.isRelative():
                                url = base_url.resolved(url)
                            filepath = url.toLocalFile()

                            if filepath and os.path.exists(filepath):
                                movie = QMovie(filepath, parent=self)
                                if movie.isValid():
                                    self._movies[name] = movie
                                    self._gif_doc_positions[name] = fragment.position()

                                    movie.jumpToFrame(0)
                                    first_frame = movie.currentPixmap()
                                    if not first_frame.isNull():
                                        self._orig_sizes[name] = first_frame.size()

                                    # Overlay QLabel on the viewport
                                    label = QLabel(self.viewport())
                                    label.setMovie(movie)
                                    label.setScaledContents(True)
                                    self._gif_labels[name] = label

                                    movie.start()
                                    label.show()
                it += 1
            block = block.next()

        # Position labels and connect scroll updates
        self._reposition_gif_labels()
        if not self._scroll_connected:
            self.verticalScrollBar().valueChanged.connect(
                self._reposition_gif_labels)
            self._scroll_connected = True

    def _reposition_gif_labels(self):
        """Move each GIF overlay label so it sits exactly on top of
        the static placeholder image in the document."""
        doc = self.document()
        doc_len = doc.characterCount()
        
        # Iterating over a copy of items to be safe
        for name, label in list(self._gif_labels.items()):
            # Safety check: ensure label hasn't been deleted
            try:
                if not label or label.isHidden():
                    continue
            except RuntimeError:
                continue

            pos = self._gif_doc_positions.get(name)
            if pos is None or pos >= doc_len:
                label.hide()
                continue

            # cursorRect gives viewport-relative coordinates
            cursor = QTextCursor(doc)
            cursor.setPosition(pos)
            rect = self.cursorRect(cursor)

            # Read current scaled dimensions from the image format
            width = height = 0
            block = doc.findBlock(pos)
            if block.isValid():
                it = block.begin()
                while not it.atEnd():
                    frag = it.fragment()
                    if (frag.isValid()
                            and frag.position() <= pos
                            < frag.position() + frag.length()):
                        f = frag.charFormat()
                        if f.isImageFormat():
                            ifmt = f.toImageFormat()
                            width = int(ifmt.width()) if ifmt.width() > 0 else 0
                            height = int(ifmt.height()) if ifmt.height() > 0 else 0
                        break
                    it += 1

            if width > 0 and height > 0:
                label.setFixedSize(width, height)
                label.move(rect.x(), rect.y())



# Ordered list of section files and their display titles
MANUAL_SECTIONS = [
    ("index.md",            "Home"),
    ("installation.md",     "Installation"),
    ("supported_data.md",   "Supported Data"),
    ("ui_overview.md",      "UI Overview"),
    ("menu_bar.md",         "Menu Bar"),
    ("spectra_maps.md",     "Workspaces Spectra & Maps"),
    ("graphs.md",           "Workspaces Graphs"),
    ("mva.md",              "Multivariate Analysis (MVA)"),
    ("calculators.md",      "Quick Calculators"),
    ("settings.md",         "Settings"),
    ("save_load.md",        "Save & Load"),
    ("shortcuts.md",        "Shortcuts & Tips"),
]


class VUserManualDialog(QDialog):
    """Dialog displaying the User Manual with a 2-panel layout:
    Left: Sections & Sub-sections (TOC)
    Right: Markdown content viewer
    (Search bar is at the top)
    """

    def __init__(self, manual_dir, parent=None):
        super().__init__(parent)
        self._text_size = 12
        self._update_stylesheet()
        
        self.manual_dir = manual_dir
        self.setWindowTitle("SPECTROview User Manual")
        self.resize(1200, 800)

        # Cache: filename -> (raw md text, html, toc_tokens)
        self._cache = {}
        self._current_file = None

        self._init_ui()
        self._load_section_list()
        # Select "Home" by default
        if self.section_list.count() > 0:
            self.section_list.setCurrentRow(0)
            self._on_section_clicked(self.section_list.item(0))

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ---- Top Bar: Search and Zoom ----
        top_bar = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Search all sections (Enter to find next)…")
        self.search_bar.textChanged.connect(self._on_search_text_changed)
        self.search_bar.returnPressed.connect(self._on_search_return_pressed)
        top_bar.addWidget(self.search_bar)
        
        top_bar.addSpacing(20)
        lbl_zoom = QLabel("Adjust Text Size:")
        top_bar.addWidget(lbl_zoom)
        
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(30)
        self.btn_zoom_out.setToolTip("Decrease Text Size")
        self.btn_zoom_out.setFocusPolicy(Qt.NoFocus)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(30)
        self.btn_zoom_in.setToolTip("Increase Text Size")
        self.btn_zoom_in.setFocusPolicy(Qt.NoFocus)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        
        top_bar.addWidget(self.btn_zoom_out)
        top_bar.addWidget(self.btn_zoom_in)
        
        layout.addLayout(top_bar)

        self.splitter = QSplitter(Qt.Horizontal)

        # ---- Left panel: Sections ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Sections")
        font = lbl.font()
        font.setBold(True)
        lbl.setFont(font)
        left_layout.addWidget(lbl)

        self.section_list = QListWidget()
        self.section_list.itemClicked.connect(self._on_section_clicked)
        left_layout.addWidget(self.section_list)

        # ---- Sub-sections (TOC) ----
        lbl2 = QLabel("Sub-sections")
        font2 = lbl2.font()
        font2.setBold(True)
        lbl2.setFont(font2)
        left_layout.addWidget(lbl2)

        self.toc_tree = QTreeWidget()
        self.toc_tree.setHeaderHidden(True)
        self.toc_tree.itemClicked.connect(self._on_toc_clicked)
        left_layout.addWidget(self.toc_tree)

        self.splitter.addWidget(left_widget)

        # ---- Center panel: Content viewer + nav buttons ----
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.content_browser = FitImageTextBrowser()
        self.content_browser.anchorClicked.connect(self._on_anchor_clicked)
        center_layout.addWidget(self.content_browser)

        # Navigation buttons
        nav_bar = QHBoxLayout()
        nav_bar.setContentsMargins(4, 2, 4, 2)
        self.btn_prev = QPushButton("← Previous")
        self.btn_prev.setFocusPolicy(Qt.NoFocus)  # Prevent focus stealing
        
        self.btn_next = QPushButton("Next →")
        self.btn_next.setFocusPolicy(Qt.NoFocus)  # Prevent focus stealing
        
        self.btn_prev.clicked.connect(self._go_previous_section)
        self.btn_next.clicked.connect(self._go_next_section)
        
        nav_bar.addWidget(self.btn_prev)
        nav_bar.addStretch()
        nav_bar.addWidget(self.btn_next)
        center_layout.addLayout(nav_bar)

        self.splitter.addWidget(center_widget)

        # Set proportions: left 250, right 950
        self.splitter.setSizes([250, 950])
        layout.addWidget(self.splitter)

    # ------------------------------------------------------------------
    # Section list population
    # ------------------------------------------------------------------
    def _load_section_list(self):
        self.section_list.clear()
        for filename, title in MANUAL_SECTIONS:
            filepath = os.path.join(self.manual_dir, filename)
            if os.path.exists(filepath):
                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, filename)
                self.section_list.addItem(item)

    # ------------------------------------------------------------------
    # Load & render a single section
    # ------------------------------------------------------------------
    def _render_section(self, filename):
        """Parse and cache a section file, then display it."""
        if filename not in self._cache:
            filepath = os.path.join(self.manual_dir, filename)
            if not os.path.exists(filepath):
                self._cache[filename] = ("", "<h3>File not found</h3>", [])
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    md_text = f.read()
                md = markdown.Markdown(
                    extensions=['toc', 'tables', 'fenced_code'])
                html = md.convert(md_text)
                toc_tokens = md.toc_tokens if hasattr(md, 'toc_tokens') else []
                self._cache[filename] = (md_text, html, toc_tokens)

        md_text, html, toc_tokens = self._cache[filename]
        self._current_file = filename

        # Set base URL so relative images (../user_manual_images/) resolve
        base_url = QUrl.fromLocalFile(
            os.path.join(self.manual_dir, "") )  # trailing sep
        self.content_browser.document().setBaseUrl(base_url)

        # Minimal adaptive CSS (no hardcoded colours)
        css = """
        <style>
            body { font-family: Verdana, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", Roboto, Helvetica, Arial;
                   line-height: 1.6; padding: 10px; }
            h1, h2, h3, h4 { margin-top: 1.2em; margin-bottom: 0.5em; }
            p { margin-top: 0; margin-bottom: 0.8em; }
            img { max-width: 100%; height: auto; }
            code { background-color: rgba(128,128,128,0.15);
                   padding: 2px 4px; border-radius: 4px;
                   font-family: "Courier New", Menlo, Consolas; }
            pre  { background-color: rgba(128,128,128,0.15);
                   padding: 10px; border-radius: 4px;
                   overflow-x: auto; margin: 1em 0; }
            pre code { background-color: transparent; padding: 0; }
            table { border-collapse: collapse; width: 100%;
                    margin: 1em 0; }
            th, td { border: 1px solid gray; padding: 8px;
                     text-align: left; }
            th { background-color: rgba(128,128,128,0.2); }
            blockquote { border-left: 4px solid gray;
                         margin: 1em 0; padding-left: 16px; }
        </style>
        """
        self.content_browser.cleanup_movies()              # stop any playing GIFs
        self.content_browser.setHtml(css + html)
        self.content_browser._resize_images()
        self.content_browser.setup_gif_animations()        # start GIF playback

        # Populate right-panel TOC
        self.toc_tree.clear()
        if toc_tokens:
            self._populate_toc(toc_tokens, self.toc_tree)
            self.toc_tree.expandAll()

        # Update prev/next navigation buttons
        self._update_nav_buttons()

    def _populate_toc(self, tokens, parent_item):
        for token in tokens:
            item = QTreeWidgetItem(parent_item)
            item.setText(0, token['name'])
            item.setData(0, Qt.UserRole, token['id'])
            if 'children' in token and token['children']:
                self._populate_toc(token['children'], item)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_section_clicked(self, item):
        filename = item.data(Qt.UserRole)
        if filename:
            self._render_section(filename)

    def _on_toc_clicked(self, item, column=0):
        anchor_id = item.data(0, Qt.UserRole)
        if anchor_id:
            self.content_browser.scrollToAnchor(anchor_id)

    def _on_anchor_clicked(self, url):
        if url.scheme() in ('http', 'https'):
            QDesktopServices.openUrl(url)
            return
        # Internal anchor
        if url.fragment():
            self.content_browser.scrollToAnchor(url.fragment())
            return
        # Could be a relative .md link (e.g. from index.md TOC)
        path = url.toLocalFile() or url.toString()
        if path.endswith('.md'):
            basename = os.path.basename(path)
            # Find and select the section
            for i in range(self.section_list.count()):
                it = self.section_list.item(i)
                if it.data(Qt.UserRole) == basename:
                    self.section_list.setCurrentItem(it)
                    self._on_section_clicked(it)
                    return

    # ------------------------------------------------------------------
    # Full-text search across ALL sections
    # ------------------------------------------------------------------
    def _on_search_text_changed(self, text):
        """Highlight the first match in the current section as the user
        types."""
        cursor = self.content_browser.textCursor()
        cursor.setPosition(0)
        self.content_browser.setTextCursor(cursor)

        if text:
            found = self.content_browser.find(text)
            if not found:
                # Not in current section – check if it exists anywhere
                self._set_search_style(not self._any_section_contains(text))
            else:
                self._set_search_style(False)
        else:
            self._set_search_style(False)

    def _on_search_return_pressed(self):
        """Pressing Enter cycles through every match across all sections."""
        text = self.search_bar.text()
        if not text:
            return

        # Ensure search bar keeps focus so the user can keep hitting Enter
        self.search_bar.setFocus()

        # Try to find the next match forward in the current section
        # Use FindFlag(0) for case-insensitive search
        found = self.content_browser.find(text, QTextDocument.FindFlag(0))
        if found:
            self._set_search_style(False)
            return

        # No more forward matches → try other sections
        current_row = self.section_list.currentRow()
        total = self.section_list.count()
        text_lower = text.lower()

        # Iterate through other sections to find the next match
        for offset in range(1, total):
            idx = (current_row + offset) % total
            item = self.section_list.item(idx)
            filename = item.data(Qt.UserRole)
            filepath = os.path.join(self.manual_dir, filename)
            
            if not os.path.exists(filepath):
                continue
                
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if text_lower not in content.lower():
                continue

            # Switch to this section
            self.section_list.setCurrentItem(item)
            self._render_section(filename)
            
            # Reset cursor to top and highlight the first match
            cursor = self.content_browser.textCursor()
            cursor.setPosition(0)
            self.content_browser.setTextCursor(cursor)
            
            if self.content_browser.find(text, QTextDocument.FindFlag(0)):
                self._set_search_style(False)
                return

        # If we exhausted all OTHER sections, wrap around within the CURRENT section
        cursor = self.content_browser.textCursor()
        cursor.setPosition(0)
        self.content_browser.setTextCursor(cursor)
        
        found = self.content_browser.find(text, QTextDocument.FindFlag(0))
        self._set_search_style(not found)

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def _any_section_contains(self, text):
        """Return True if *any* section file contains `text`."""
        text_lower = text.lower()
        for filename, _ in MANUAL_SECTIONS:
            filepath = os.path.join(self.manual_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    if text_lower in f.read().lower():
                        return True
        return False

    # ------------------------------------------------------------------
    # Previous / Next section navigation
    # ------------------------------------------------------------------
    def _go_previous_section(self):
        current = self.section_list.currentRow()
        if current > 0:
            self.section_list.setCurrentRow(current - 1)
            self._on_section_clicked(self.section_list.item(current - 1))

    def _go_next_section(self):
        current = self.section_list.currentRow()
        if current < self.section_list.count() - 1:
            self.section_list.setCurrentRow(current + 1)
            self._on_section_clicked(self.section_list.item(current + 1))

    def _update_nav_buttons(self):
        """Enable/disable prev/next buttons based on current position."""
        current = self.section_list.currentRow()
        total = self.section_list.count()
        self.btn_prev.setEnabled(current > 0)
        self.btn_next.setEnabled(current < total - 1)
        
        # Show the target section name in the button text
        if current > 0:
            prev_title = self.section_list.item(current - 1).text()
            self.btn_prev.setText(f"← {prev_title}")
        else:
            self.btn_prev.setText("← Previous")
            
        if current < total - 1:
            next_title = self.section_list.item(current + 1).text()
            self.btn_next.setText(f"{next_title} →")
        else:
            self.btn_next.setText("Next →")

    def _set_search_style(self, not_found):
        if not_found:
            self.search_bar.setStyleSheet(
                "QLineEdit { background-color: #ffcccc; color: black; }")
        else:
            self.search_bar.setStyleSheet("")

    def _zoom_in(self):
        self._text_size += 1
        self._update_stylesheet()

    def _zoom_out(self):
        self._text_size = max(6, self._text_size - 1)
        self._update_stylesheet()

    def _update_stylesheet(self):
        self.setStyleSheet(f"""
            QWidget {{
                font-family: Verdana, Arial;
                font-size: 12pt;
            }}
            QTextBrowser {{
                font-size: {self._text_size}pt;
            }}
        """)
