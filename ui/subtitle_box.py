import html
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics, QTextOption
from PySide6.QtWidgets import QTextBrowser, QWidget


class SubtitleBox(QTextBrowser):
    """A centered caption box with a bounded height for live display."""

    def __init__(self, target_width: int, is_current: bool = False, parent: QWidget | None = None, *, max_lines=2) -> None:
        super().__init__(parent)
        self.target_width = target_width
        self.is_current = is_current
        self.max_lines = max_lines
        self.is_final: bool = True
        self._last_content = None
        self._layout_width = None
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.apply_box_style(is_final=True)

    def apply_box_style(self, is_final: bool) -> None:
        self.is_final = is_final
        if not self.is_current:
            self.setStyleSheet("""
                QTextBrowser {
                    color: #D1D5DB;
                    background-color: rgba(24, 24, 24, 210);
                    border: 1px solid #383838;
                    border-radius: 8px;
                    padding: 10px 20px;
                }
            """)
            return

        if is_final:
            self.setStyleSheet("""
                QTextBrowser {
                    color: #FFE600;
                    background-color: rgba(24, 24, 24, 242);
                    border: 2px solid #FFE600;
                    border-radius: 10px;
                    padding: 12px 24px;
                }
            """)
        else:
            self.setStyleSheet("""
                QTextBrowser {
                    color: #FFFFFF;
                    background-color: rgba(20, 28, 42, 242);
                    border: 2px dashed #60A5FA;
                    border-radius: 10px;
                    padding: 12px 24px;
                }
            """)

    def apply_width_for_parent(self, parent_width: int) -> None:
        usable = max(320, min(self.target_width, parent_width - 80))
        if usable == self._layout_width:
            return
        self._layout_width = usable
        self.setFixedWidth(usable)
        self.sync_height()

    def set_subtitle_text(self, text: str, font_size: int, is_final: bool = True) -> None:
        content = (text, font_size, is_final)
        if content == self._last_content:
            return
        self._last_content = content
        self.apply_box_style(is_final)

        font = QFont("Malgun Gothic")
        font.setPixelSize(font_size)
        font.setItalic(False)
        font.setBold(self.is_current and is_final)
        self.setFont(font)
        self.document().setDefaultFont(font)

        self.setHtml(f'<div align="center">{html.escape(text)}</div>')
        self.sync_height()

    def sync_height(self) -> None:
        doc = self.document()
        horizontal_padding = 48 if self.is_current else 40
        doc.setTextWidth(max(200, self.width() - horizontal_padding))
        doc_height = int(doc.size().height())
        vertical_padding = 28 if self.is_current else 24
        max_height = QFontMetrics(doc.defaultFont()).lineSpacing() * self.max_lines + 8
        self.setFixedHeight(max(42, min(doc_height, max_height) + vertical_padding))
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
