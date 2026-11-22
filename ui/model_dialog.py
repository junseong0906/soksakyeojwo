from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QPushButton, QProgressBar, QCheckBox, QComboBox, QLineEdit,
    QAbstractItemView, QTextEdit, QButtonGroup, QWidget, QApplication)
from model_catalog import MODELS, get_model
from model_service import human_bytes
from app_version import APP_VERSION


class ModelDialog(QDialog):
    applyRequested = Signal(str, bool, str, str)
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle(f'설정 · 음성 인식 모델 v{APP_VERSION}')
        screen = self.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry()
        self.setMinimumSize(min(930, available.width() - 48), min(600, available.height() - 48))
        self.resize(min(1140, available.width() - 48), min(850, available.height() - 48))
        self.setStyleSheet('''
            QDialog {background:#101827; color:#f3f6fc;}
            QLabel,QCheckBox {color:#e5edf9;font-size:14px;}
            QTableWidget {background:#172132;color:#edf4ff;gridline-color:#344154;selection-background-color:#203f6b;font-size:15px;border:1px solid #344154;}
            QHeaderView::section {background:#222e40;color:#cbd7e8;border:0;padding:8px;font-weight:600;}
            QTableWidget::item {padding:5px;}
            QCheckBox::indicator {width:18px;height:18px;}
            QPushButton {background:#243249;color:#f5f8ff;border:1px solid #41506a;border-radius:7px;padding:10px 17px;}
            QPushButton:disabled {color:#778399;background:#1b2535;}
            QPushButton#primary {background:#2563eb;border-color:#5592ff;}
            QLineEdit,QComboBox,QTextEdit {background:#1a2638;color:#edf4ff;border:1px solid #40516d;border-radius:5px;padding:6px;}
            QProgressBar {background:#202c40;color:#fff;border:1px solid #40516d;border-radius:5px;text-align:center;min-height:20px;}
            QProgressBar::chunk {background:#3b82f6;border-radius:4px;}
        ''')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        title = QLabel('음성 인식 모델')
        title.setStyleSheet('font-size:26px;font-weight:700;')
        layout.addWidget(title)
        self.current = QLabel()
        layout.addWidget(self.current)
        self.table = QTableWidget(len(MODELS), 6)
        self.table.setHorizontalHeaderLabels(['적용 선택', '모델 · 한국어 지원', '용량', '상태', '다운로드', '진행률'])
        header = self.table.horizontalHeader()
        header.setFixedHeight(42)
        for column in (0, 2, 3, 4): header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(5, 126)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.choices = QButtonGroup(self)
        self.choices.setExclusive(True)
        self.model_checks = []
        self.download_buttons = []
        self.row_progress = []
        for i, model in enumerate(MODELS):
            self.table.setRowHeight(i, 54)
            check = QCheckBox()
            check.setAccessibleName(model.name + ' 적용 선택')
            check.setToolTip('적용할 모델은 한 번에 하나만 선택합니다.')
            self.choices.addButton(check, i)
            self.model_checks.append(check)
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.addWidget(check, alignment=Qt.AlignmentFlag.AlignCenter)
            self.table.setCellWidget(i, 0, cell)
            self.table.setItem(i, 1, QTableWidgetItem(model.name))
            self.table.setItem(i, 2, QTableWidgetItem(model.size_hint))
            self.table.setItem(i, 3, QTableWidgetItem('미설치'))
            download = QPushButton('다운로드')
            download.setAccessibleName(model.name + ' 다운로드')
            download.clicked.connect(lambda checked=False, model_id=model.id: self._download_model(model_id))
            self.download_buttons.append(download)
            self.table.setCellWidget(i, 4, download)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat('%p%')
            self.row_progress.append(bar)
            self.table.setCellWidget(i, 5, bar)
        self.table.setMinimumHeight(200)
        layout.addWidget(self.table, 1)
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setStyleSheet('background:#202c3e;padding:13px;border-radius:8px;')
        layout.addWidget(self.description)
        options = QHBoxLayout()
        self.diarization = QCheckBox('화자 분리 켜기')
        self.diarization.setChecked(bool(controller.settings.value['diarization']))
        options.addWidget(self.diarization)
        options.addStretch()
        options.addWidget(QLabel('실행 장치'))
        self.device = QComboBox()
        self.device.addItem('자동 (CPU 우선)', 'auto')
        self.device.addItem('CPU', 'cpu')
        self.device.addItem('NVIDIA GPU (CUDA)', 'cuda')
        self.device.setCurrentIndex(max(0, self.device.findData(controller.settings.value['device'])))
        options.addWidget(self.device)
        layout.addLayout(options)
        token_row = QHBoxLayout()
        token_row.addWidget(QLabel('화자 모델 접근 토큰'))
        self.token = QLineEdit(controller.token)
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText('Hugging Face 읽기 토큰 · 이번 실행에서만 사용')
        token_row.addWidget(self.token)
        layout.addLayout(token_row)
        guide = QLabel('화자 분리: <a style="color:#93c5fd" href="https://huggingface.co/pyannote/segmentation-3.0">segmentation-3.0</a> 및 <a style="color:#93c5fd" href="https://huggingface.co/pyannote/embedding">embedding</a> 이용 동의와 <a style="color:#93c5fd" href="https://huggingface.co/settings/tokens">읽기 토큰</a>이 필요합니다. 화자 ID는 연결마다 달라질 수 있습니다.')
        guide.setOpenExternalLinks(True)
        guide.setWordWrap(True)
        guide.setStyleSheet('font-size:12px;color:#aebfd7;')
        layout.addWidget(guide)
        self.status = QLabel(controller.status)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(90)
        self.details.hide()
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        self.prepare = QPushButton('실행환경 준비')
        self.cancel = QPushButton('중지')
        self.apply = QPushButton('선택한 모델 적용')
        self.apply.setObjectName('primary')
        close = QPushButton('닫기')
        self.prepare.clicked.connect(self._prepare)
        self.cancel.clicked.connect(controller.stop_task)
        self.apply.clicked.connect(self._apply)
        close.clicked.connect(self.accept)
        for button in (self.prepare, self.cancel): actions.addWidget(button)
        actions.addStretch()
        actions.addWidget(self.apply)
        actions.addWidget(close)
        layout.addLayout(actions)
        footer = QHBoxLayout()
        folder = QPushButton('모델 폴더 열기')
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(controller.models.root))))
        logs = QPushButton('로그 폴더 열기')
        from app_paths import data_dir
        logs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(data_dir() / 'logs'))))
        footer_hint = QLabel('7종 · 각 행에서 다운로드한 뒤, 체크박스로 모델 하나를 선택하고 적용하세요.')
        footer_hint.setWordWrap(True)
        footer.addWidget(footer_hint, 1)
        footer.addWidget(logs)
        footer.addWidget(folder)
        layout.addLayout(footer)
        self.table.itemSelectionChanged.connect(self._row_selected)
        self.choices.idToggled.connect(self._choice_changed)
        self.diarization.toggled.connect(self.refresh)
        self.device.currentIndexChanged.connect(self.refresh)
        controller.changed.connect(self.refresh)
        controller.progress.connect(self._progress)
        controller.failed.connect(self._error)
        row = next((i for i, model in enumerate(MODELS) if model.id == controller.settings.value['model_id']), 0)
        self.model_checks[row].setChecked(True)
        self.refresh()

    def selected(self): return MODELS[max(0, self.choices.checkedId())]
    def _row_selected(self):
        if self.controller.busy or getattr(self.parent(), 'pending_model', None): return
        row = self.table.currentRow()
        if row >= 0: self.model_checks[row].setChecked(True)
    def _choice_changed(self, row, checked):
        if checked:
            self.table.selectRow(row)
            self.refresh()
    def refresh(self, *_):
        controller = self.controller
        model = self.selected()
        busy = controller.busy or bool(getattr(self.parent(), 'pending_model', None))
        active = get_model(controller.active_id).name if controller.active_id else '엔진 미적용'
        self.current.setText(f'현재 사용: {active} · 언어: 한국어')
        self.description.setText('적용 선택: ' + model.name + ' · ' + model.description + '\n' + model.repo)
        for i, item in enumerate(MODELS):
            installed = controller.models.installed(item)
            ready = controller.runtimes.ready(item, self.diarization.isChecked(), self.device.currentData())
            job = controller.download_states.get(item.id, {})
            state = '사용 중' if item.id == controller.active_id else ('다운로드 완료' if installed else job.get('state', '미설치'))
            self.table.item(i, 3).setText(state)
            self.table.item(i, 3).setToolTip('실행환경 준비 완료' if ready else '적용 시 필요한 실행환경을 준비합니다.')
            size = controller.models.size(item)
            self.table.item(i, 2).setText(human_bytes(size) if size is not None else item.size_hint)
            active_download = controller.busy and controller.task_kind == 'download' and controller.task_model_id == item.id
            button = self.download_buttons[i]
            button.setEnabled(not busy)
            button.setText('다운로드 중' if active_download else ('다시 확인' if installed else ('다시 시도' if job.get('state') in ('다운로드 실패', '중지됨') else '다운로드')))
            bar = self.row_progress[i]
            percent = 100 if installed else job.get('percent', 0)
            bar.setRange(0, 0 if active_download and percent < 0 else 100)
            if percent >= 0: bar.setValue(percent)
            self.model_checks[i].setEnabled(not busy)
        self.prepare.setEnabled(not busy)
        self.apply.setEnabled(not busy and controller.models.installed(model))
        self.cancel.setEnabled(controller.busy)
        self.table.setEnabled(True)
        self.diarization.setEnabled(not busy)
        self.device.setEnabled(not busy)
        self.token.setEnabled(not busy)
        self.progress.setVisible(controller.busy and controller.task_kind != 'download')
    def _download_model(self, model_id):
        self.details.hide()
        self.controller.download(model_id, self.token.text())
    def _prepare(self):
        self.details.hide()
        self.controller.prepare(self.selected().id, self.diarization.isChecked(), self.device.currentData())
    def _apply(self):
        self.details.hide()
        self.applyRequested.emit(self.selected().id, self.diarization.isChecked(), self.device.currentData(), self.token.text())
        self.refresh()
    def _progress(self, percent, message):
        controller = self.controller
        prefix = ''
        if controller.task_kind == 'download' and controller.task_model_id:
            row = next(i for i, model in enumerate(MODELS) if model.id == controller.task_model_id)
            prefix = MODELS[row].name + ' · '
            self.table.item(row, 3).setText(controller.download_states.get(controller.task_model_id, {}).get('state', '다운로드 중'))
            bar = self.row_progress[row]
            bar.setRange(0, 0 if percent < 0 else 100)
            if percent >= 0: bar.setValue(percent)
        self.status.setText(prefix + message.split('\n')[0][:240])
        self.progress.setVisible(controller.busy and controller.task_kind != 'download')
        self.progress.setRange(0, 0 if percent < 0 else 100)
        if percent >= 0: self.progress.setValue(percent)
    def _error(self, message):
        self.details.setPlainText(message)
        self.details.show()
        self.refresh()
