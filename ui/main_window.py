from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QShortcut,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ui.subtitle_box import SubtitleBox
from workers.asr_client_worker import asr_client_process_main
from workers.recorder_worker import recorder_process_main

SAMPLE_STREAM: list[tuple[str, bool]] = [
    ("오늘은 음성 신호 처리와 실시간", False),
    ("오늘은 음성 신호 처리와 실시간 자막 파이프라인 구조를 살펴보겠습니다.", True),
    ("마이크 입력이 들어오면 녹음 경로와", False),
    ("마이크 입력이 들어오면 녹음 경로와 자막 처리 경로를 분리하는 것이 핵심입니다.", True),
    ("중요한 개념은 다시 확인할 수 있도록", False),
    ("중요한 개념은 다시 확인할 수 있도록 북마크를 남겨두는 게 좋습니다.", True),
    ("자막이 잠시 멈추거나 지연되더라도", False),
    ("자막이 잠시 멈추거나 지연되더라도 녹음은 계속 이어지도록 설계합니다.", True),
]


class SpeakerLabeler:
    def __init__(self) -> None:
        self._speaker_order: dict[str, int] = {}
        self._custom_names: dict[str, str] = {}

    def set_custom_name(self, speaker_id: str, display_name: str) -> None:
        sid = speaker_id.strip()
        if not sid:
            return
        name = display_name.strip()
        if name:
            self._custom_names[sid] = name
        else:
            self._custom_names.pop(sid, None)

    def get_custom_name(self, speaker_id: str) -> str:
        return self._custom_names.get(speaker_id.strip(), "")

    def apply(self, ev: dict) -> dict:
        sid = str(ev.get("speaker_id") or "").strip()
        if not sid:
            return {**ev, "speaker_label": ""}
        if sid in self._custom_names:
            label = self._custom_names[sid]
        else:
            if sid not in self._speaker_order:
                self._speaker_order[sid] = len(self._speaker_order) + 1
            label = f"화자 {self._speaker_order[sid]}"
        return {**ev, "speaker_label": label}


class SpeakerLabelDialog(QDialog):
    def __init__(self, labeler: SpeakerLabeler, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.labeler = labeler
        self.setWindowTitle("화자 이름 설정")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.speaker_id_input = QLineEdit()
        self.speaker_id_input.setPlaceholderText("예: spk_1 또는 1")
        self.speaker_id_input.textChanged.connect(self._on_id_changed)

        self.display_name_input = QLineEdit()
        self.display_name_input.setPlaceholderText("비워두면 기본 라벨 사용")

        self.current_map_label = QLabel("현재 지정됨: 없음")

        form.addRow("화자 ID", self.speaker_id_input)
        form.addRow("표시 이름", self.display_name_input)
        form.addRow("상태", self.current_map_label)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton("적용")
        apply_btn.clicked.connect(self._apply)
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.reject)
        btn_row.addStretch(1)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _on_id_changed(self, text: str) -> None:
        sid = text.strip()
        existing = self.labeler.get_custom_name(sid) if sid else ""
        self.display_name_input.setText(existing)
        self.current_map_label.setText(
            f"현재 지정됨: {sid} → {existing}" if (sid and existing) else "현재 지정됨: 없음"
        )

    def _apply(self) -> None:
        sid = self.speaker_id_input.text().strip()
        if not sid:
            return
        name = self.display_name_input.text().strip()
        self.labeler.set_custom_name(sid, name)
        self._on_id_changed(sid)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("속삭여줘v1")
        self.resize(1240, 840)

        self.bookmarks: list[str] = []
        self.history_entries: list[str] = []
        self.font_size: int = 28
        self.min_font_size: int = 18
        self.max_font_size: int = 56
        self.sample_index: int = 0
        self.is_playing_sample: bool = False
        self.is_recording: bool = False
        self.is_live_asr_active: bool = False
        self.is_focus_mode: bool = False
        self.saved_file_path: str = ""

        # 이전 자막 탐색 & 검색 상태
        self.auto_scroll: bool = True
        self.unread_new_count: int = 0
        self.search_spans: list[tuple[int, int]] = []
        self.search_cursor_idx: int = -1
        self.last_search_query: str = ""

        # 자막 상태
        self.sub_mode: str = "대기 중"
        self.record_only_seconds: int = 0
        self.alert_threshold_sec: int = 30

        self.prev_text: str = ""
        self.current_text: str = "WLK 서버로 실시간 자막을 시작하거나 「테스트 자막 재생」을 눌러보세요"
        self.current_is_final: bool = True
        self.speaker_labeler = SpeakerLabeler()
        self.current_speaker_label: str = ""
        self.prev_speaker_label: str = ""
        self.last_committed_raw: str = ""
        self.last_committed_speaker: str = ""

        self._build_ui()
        self._init_shortcuts_and_timers()
        self._start_workers()
        self.refresh_subtitles()

    def open_speaker_label_dialog(self) -> None:
        dlg = SpeakerLabelDialog(self.speaker_labeler, self)
        dlg.exec()

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
                color: #FFFFFF;
            }
            QWidget#topBar, QWidget#micBar, QWidget#simBar, QWidget#historyBar {
                background-color: #121212;
                border-bottom: 1px solid #333333;
            }
            QLabel#recStatus {
                color: #4ADE80;
                background-color: #052E16;
                border: 1px solid #16A34A;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QLabel#subStatus {
                color: #93C5FD;
                background-color: #172554;
                border: 1px solid #2563EB;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QLabel#criticalAlert {
                color: #FFFFFF;
                background-color: #991B1B;
                border: 2px solid #F87171;
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QLabel#stateBadge {
                color: #93C5FD;
                background-color: #1E3A8A;
                border: 1px solid #60A5FA;
                border-radius: 5px;
                padding: 4px 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QLabel#toastBadge {
                color: #000000;
                background-color: #FFE600;
                border-radius: 5px;
                padding: 5px 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QLabel#scrollPausedBadge {
                color: #FDE047;
                background-color: #422006;
                border: 1px solid #EAB308;
                border-radius: 5px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton {
                color: #FFFFFF;
                background-color: #262626;
                border: 1px solid #525252;
                border-radius: 6px;
                padding: 7px 12px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #383838;
            }
            QPushButton#asrBtn {
                color: #FFFFFF;
                background-color: #1D4ED8;
                border: 1px solid #60A5FA;
            }
            QPushButton#asrBtn:hover {
                background-color: #2563EB;
            }
            QPushButton#recBtn {
                color: #FFFFFF;
                background-color: #15803D;
                border: 1px solid #22C55E;
            }
            QPushButton#recBtn:hover {
                background-color: #16A34A;
            }
            QPushButton#focusBtn {
                color: #E0F2FE;
                background-color: #0C4A6E;
                border: 1px solid #38BDF8;
            }
            QPushButton#focusBtn:hover {
                background-color: #0369A1;
            }
            QPushButton#returnLatestBtn {
                color: #000000;
                background-color: #38BDF8;
                border: none;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton#returnLatestBtn:hover {
                background-color: #7DD3FC;
            }
            QPushButton#bookmarkBtn {
                color: #000000;
                background-color: #FFE600;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton#bookmarkBtn:hover {
                background-color: #FFF04D;
            }
            QLineEdit#searchInput, QLineEdit#urlInput {
                color: #FFFFFF;
                background-color: #1A1A1A;
                border: 1px solid #525252;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }
            QLineEdit#searchInput {
                min-width: 190px;
            }
            QLineEdit#urlInput {
                min-width: 210px;
            }
            QTextBrowser#historyBrowser {
                color: #F3F4F6;
                background-color: #0B0B0B;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 15px;
            }
            QComboBox {
                color: #FFFFFF;
                background-color: #1E1E1E;
                border: 1px solid #525252;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
                min-width: 220px;
            }
            QComboBox QAbstractItemView {
                color: #FFFFFF;
                background-color: #1E1E1E;
                selection-background-color: #2563EB;
            }
            QProgressBar {
                background-color: #1E1E1E;
                border: 1px solid #444444;
                border-radius: 5px;
                text-align: center;
                color: #FFFFFF;
                font-size: 12px;
                height: 18px;
                min-width: 120px;
            }
            QProgressBar::chunk {
                background-color: #22C55E;
                border-radius: 4px;
            }
            QLabel#infoText {
                color: #E5E7EB;
                font-size: 13px;
            }
        """)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 상단 바
        self.top_bar = QWidget()
        self.top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(20, 10, 20, 10)
        top_layout.setSpacing(8)

        self.rec_status_label = QLabel("녹음: 대기 중")
        self.rec_status_label.setObjectName("recStatus")

        self.sub_status_label = QLabel("자막: 대기 중")
        self.sub_status_label.setObjectName("subStatus")

        self.font_smaller_btn = QPushButton("글씨 -")
        self.font_smaller_btn.clicked.connect(self.decrease_font)

        self.font_larger_btn = QPushButton("글씨 +")
        self.font_larger_btn.clicked.connect(self.increase_font)

        self.font_size_label = QLabel(f"{self.font_size}px")
        self.font_size_label.setObjectName("infoText")

        self.live_asr_btn = QPushButton("실시간 자막 시작")
        self.live_asr_btn.setObjectName("asrBtn")
        self.live_asr_btn.clicked.connect(self.toggle_live_asr)

        self.sample_btn = QPushButton("테스트 자막 재생")
        self.sample_btn.clicked.connect(self.toggle_sample_subtitles)

        self.focus_btn = QPushButton("자막 집중 화면")
        self.focus_btn.setObjectName("focusBtn")
        self.focus_btn.clicked.connect(self.toggle_focus_mode)

        self.bookmark_info_btn = QPushButton("북마크 0개")
        self.bookmark_info_btn.clicked.connect(self.show_bookmarks_dialog)

        self.bookmark_btn = QPushButton("이 부분 다시 확인")
        self.bookmark_btn.setObjectName("bookmarkBtn")
        self.bookmark_btn.clicked.connect(self.add_bookmark)

        top_layout.addWidget(self.rec_status_label)
        top_layout.addWidget(self.sub_status_label)
        top_layout.addSpacing(4)
        top_layout.addWidget(self.font_smaller_btn)
        top_layout.addWidget(self.font_size_label)
        top_layout.addWidget(self.font_larger_btn)
        top_layout.addSpacing(4)
        top_layout.addWidget(self.live_asr_btn)
        top_layout.addWidget(self.sample_btn)
        top_layout.addWidget(self.focus_btn)
        top_layout.addStretch(1)
        top_layout.addWidget(self.bookmark_info_btn)
        top_layout.addWidget(self.bookmark_btn)
        main_layout.addWidget(self.top_bar)

        # 2. 마이크 & WLK 주소 바
        self.mic_bar = QWidget()
        self.mic_bar.setObjectName("micBar")
        mic_layout = QHBoxLayout(self.mic_bar)
        mic_layout.setContentsMargins(20, 8, 20, 8)
        mic_layout.setSpacing(8)

        mic_title = QLabel("입력 마이크:")
        mic_title.setObjectName("infoText")

        self.mic_combo = QComboBox()
        self.mic_combo.currentIndexChanged.connect(self.on_device_changed)

        self.refresh_dev_btn = QPushButton("장치 새로고침")
        self.refresh_dev_btn.clicked.connect(self.request_refresh_devices)

        self.rec_toggle_btn = QPushButton("녹음 시작")
        self.rec_toggle_btn.setObjectName("recBtn")
        self.rec_toggle_btn.clicked.connect(self.toggle_recording)

        level_title = QLabel("레벨:")
        level_title.setObjectName("infoText")

        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setFormat("0%")

        url_title = QLabel("WLK:")
        url_title.setObjectName("infoText")

        self.wlk_url_input = QLineEdit("ws://127.0.0.1:8000/asr")
        self.wlk_url_input.setObjectName("urlInput")

        self.speaker_btn = QPushButton("화자 이름 설정")
        self.speaker_btn.clicked.connect(self.open_speaker_label_dialog)

        self.focus_hint_label = QLabel("집중 화면 (Esc 또는 버튼으로 해제)")
        self.focus_hint_label.setObjectName("infoText")
        self.focus_hint_label.hide()

        mic_layout.addWidget(mic_title)
        mic_layout.addWidget(self.mic_combo)
        mic_layout.addWidget(self.refresh_dev_btn)
        mic_layout.addWidget(self.rec_toggle_btn)
        mic_layout.addSpacing(6)
        mic_layout.addWidget(level_title)
        mic_layout.addWidget(self.level_bar)
        mic_layout.addSpacing(6)
        mic_layout.addWidget(url_title)
        mic_layout.addWidget(self.wlk_url_input)
        mic_layout.addWidget(self.speaker_btn)
        mic_layout.addStretch(1)
        mic_layout.addWidget(self.focus_hint_label)
        main_layout.addWidget(self.mic_bar)

        # 3. 상태/장애 테스트 바
        self.sim_bar = QWidget()
        self.sim_bar.setObjectName("simBar")
        sim_layout = QHBoxLayout(self.sim_bar)
        sim_layout.setContentsMargins(20, 7, 20, 7)
        sim_layout.setSpacing(8)

        sim_title = QLabel("상태/장애 테스트:")
        sim_title.setObjectName("infoText")

        self.btn_delay = QPushButton("자막 지연")
        self.btn_delay.clicked.connect(self.set_delayed_mode)

        self.btn_record_only = QPushButton("자막 중단·녹음 유지")
        self.btn_record_only.clicked.connect(self.set_record_only_mode)

        self.btn_30s_test = QPushButton("30초 초과 경고 보기")
        self.btn_30s_test.clicked.connect(self.trigger_30s_alert_test)

        self.btn_normal = QPushButton("정상 복귀")
        self.btn_normal.clicked.connect(self.restore_normal_mode)

        self.record_only_counter_label = QLabel("중단 경과: 0초 / 30초")
        self.record_only_counter_label.setObjectName("infoText")

        sim_layout.addWidget(sim_title)
        sim_layout.addWidget(self.btn_delay)
        sim_layout.addWidget(self.btn_record_only)
        sim_layout.addWidget(self.btn_30s_test)
        sim_layout.addWidget(self.btn_normal)
        sim_layout.addSpacing(10)
        sim_layout.addWidget(self.record_only_counter_label)
        sim_layout.addStretch(1)
        main_layout.addWidget(self.sim_bar)

        # 4. 심각 경고 배너
        self.critical_alert_banner = QLabel("")
        self.critical_alert_banner.setObjectName("criticalAlert")
        self.critical_alert_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.critical_alert_banner.hide()
        main_layout.addWidget(self.critical_alert_banner)

        # 5. 이전 자막 탐색 & 검색 영역
        self.history_section = QWidget()
        history_layout = QVBoxLayout(self.history_section)
        history_layout.setContentsMargins(24, 12, 24, 8)
        history_layout.setSpacing(8)

        self.history_bar = QWidget()
        self.history_bar.setObjectName("historyBar")
        hbar_layout = QHBoxLayout(self.history_bar)
        hbar_layout.setContentsMargins(8, 6, 8, 6)
        hbar_layout.setSpacing(8)

        history_title = QLabel("이전 자막 기록:")
        history_title.setObjectName("infoText")

        self.scroll_paused_badge = QLabel("자동스크롤 정지")
        self.scroll_paused_badge.setObjectName("scrollPausedBadge")
        self.scroll_paused_badge.hide()

        self.return_latest_btn = QPushButton("새 자막 0개 · 최신으로")
        self.return_latest_btn.setObjectName("returnLatestBtn")
        self.return_latest_btn.clicked.connect(self.jump_to_latest)
        self.return_latest_btn.hide()

        self.search_count_label = QLabel("")
        self.search_count_label.setObjectName("infoText")

        self.search_input = QLineEdit()
        self.search_input.setObjectName("searchInput")
        self.search_input.setPlaceholderText("이전 자막 검색...")
        self.search_input.textChanged.connect(self.on_search_text_changed)
        self.search_input.returnPressed.connect(self.find_next_match)

        self.search_prev_btn = QPushButton("이전 찾기")
        self.search_prev_btn.clicked.connect(self.find_prev_match)

        self.search_next_btn = QPushButton("다음 찾기")
        self.search_next_btn.clicked.connect(self.find_next_match)

        hbar_layout.addWidget(history_title)
        hbar_layout.addWidget(self.scroll_paused_badge)
        hbar_layout.addWidget(self.return_latest_btn)
        hbar_layout.addStretch(1)
        hbar_layout.addWidget(self.search_count_label)
        hbar_layout.addWidget(self.search_input)
        hbar_layout.addWidget(self.search_prev_btn)
        hbar_layout.addWidget(self.search_next_btn)

        self.history_browser = QTextBrowser()
        self.history_browser.setObjectName("historyBrowser")
        self.history_browser.setReadOnly(True)
        self.history_browser.setFont(QFont("Malgun Gothic", 11))
        self.history_browser.setMaximumHeight(165)
        self.history_browser.verticalScrollBar().valueChanged.connect(self.on_history_scroll_changed)

        history_layout.addWidget(self.history_bar)
        history_layout.addWidget(self.history_browser)
        main_layout.addWidget(self.history_section)

        # 6. 영화 자막 영역(가운데 아래 배치)
        self.body_widget = QWidget()
        body_layout = QVBoxLayout(self.body_widget)
        body_layout.setContentsMargins(40, 12, 40, 32)
        body_layout.setSpacing(10)
        body_layout.addStretch(1)

        self.toast_badge = QLabel("")
        self.toast_badge.setObjectName("toastBadge")
        self.toast_badge.hide()

        self.state_badge = QLabel("인식 중")
        self.state_badge.setObjectName("stateBadge")
        self.state_badge.hide()

        self.prev_subtitle_box = SubtitleBox(target_width=880, is_current=False)
        self.prev_subtitle_box.hide()

        self.current_subtitle_box = SubtitleBox(target_width=940, is_current=True)

        body_layout.addWidget(self.toast_badge, alignment=Qt.AlignmentFlag.AlignHCenter)
        body_layout.addWidget(self.state_badge, alignment=Qt.AlignmentFlag.AlignHCenter)
        body_layout.addWidget(self.prev_subtitle_box, alignment=Qt.AlignmentFlag.AlignHCenter)
        body_layout.addWidget(self.current_subtitle_box, alignment=Qt.AlignmentFlag.AlignHCenter)

        main_layout.addWidget(self.body_widget, stretch=1)
        self.setCentralWidget(central)

    def _init_shortcuts_and_timers(self) -> None:
        esc_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc_shortcut.activated.connect(self.exit_focus_mode)

        bm_shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        bm_shortcut.activated.connect(self.add_bookmark)

        self.sample_timer = QTimer(self)
        self.sample_timer.setInterval(1800)
        self.sample_timer.timeout.connect(self.push_next_sample)

        self.toast_timer = QTimer(self)
        self.toast_timer.setSingleShot(True)
        self.toast_timer.setInterval(2500)
        self.toast_timer.timeout.connect(self.toast_badge.hide)

        self.record_only_timer = QTimer(self)
        self.record_only_timer.setInterval(1000)
        self.record_only_timer.timeout.connect(self.tick_record_only)

        self.worker_poll_timer = QTimer(self)
        self.worker_poll_timer.setInterval(50)
        self.worker_poll_timer.timeout.connect(self._poll_worker_events)

    def _start_workers(self) -> None:
        self.rec_cmd_queue: Queue = Queue()
        self.rec_event_queue: Queue = Queue()
        self.asr_cmd_queue: Queue = Queue()
        self.asr_event_queue: Queue = Queue()

        self.recorder_proc = Process(
            target=recorder_process_main,
            args=(self.rec_cmd_queue, self.rec_event_queue, self.asr_cmd_queue),
            daemon=True,
        )
        self.asr_proc = Process(
            target=asr_client_process_main,
            args=(self.asr_cmd_queue, self.asr_event_queue),
            daemon=True,
        )
        self.recorder_proc.start()
        self.asr_proc.start()
        self.worker_poll_timer.start()

    def _show_toast(self, msg: str) -> None:
        self.toast_badge.setText(msg)
        self.toast_badge.show()
        self.toast_timer.start()

    def _poll_worker_events(self) -> None:
        while not self.rec_event_queue.empty():
            ev = self.rec_event_queue.get_nowait()
            etype = ev.get("type")

            if etype == "devices":
                devs = ev.get("devices", [])
                selected = ev.get("selected")
                self.mic_combo.blockSignals(True)
                self.mic_combo.clear()
                target_combo_idx = 0
                for i, d in enumerate(devs):
                    self.mic_combo.addItem(str(d["label"]), d["index"])
                    if d["index"] == selected:
                        target_combo_idx = i
                if devs:
                    self.mic_combo.setCurrentIndex(target_combo_idx)
                else:
                    self.mic_combo.addItem("사용 가능한 입력 마이크 없음", None)
                self.mic_combo.blockSignals(False)

            elif etype == "level":
                val = int(ev.get("value", 0))
                self.level_bar.setValue(val)
                self.level_bar.setFormat(f"{val}%")

            elif etype == "recording_started":
                self.is_recording = True
                self.saved_file_path = str(ev.get("file", ""))
                fname = Path(self.saved_file_path).name if self.saved_file_path else "저장 중"
                self.rec_toggle_btn.setText("녹음 정지")
                self.rec_status_label.setText(f"녹음: 저장 중 ({fname})")
                self._show_toast(f"로컬 녹음 시작: recordings/{fname}")

            elif etype == "recording_stopped":
                self.is_recording = False
                fname = Path(self.saved_file_path).name if self.saved_file_path else "완료"
                self.rec_toggle_btn.setText("녹음 시작")
                self.rec_status_label.setText("녹음: 대기 중")
                self._show_toast(f"녹음 저장 완료: {fname}")

            elif etype == "error":
                self._show_toast(str(ev.get("message", "오디오 오류")))

        while not self.asr_event_queue.empty():
            ev = self.asr_event_queue.get_nowait()
            etype = ev.get("type")

            if etype == "stream_opened":
                self.is_live_asr_active = True
                self.sub_mode = "수신 중"
                self.sub_status_label.setText("자막: WLK 연결됨")
                self.live_asr_btn.setText("자막 일시정지")
                self.critical_alert_banner.hide()
                self.record_only_timer.stop()
                self.record_only_seconds = 0
                self.record_only_counter_label.setText("중단 경과: 0초 / 30초")

            elif etype == "stream_closed":
                self.is_live_asr_active = False
                self.live_asr_btn.setText("실시간 자막 재개")
                if self.sub_mode != "중단(녹음 유지)":
                    self.sub_mode = "대기 중"
                    self.sub_status_label.setText("자막: 일시정지/대기")
                self.current_is_final = True
                self.refresh_subtitles()

            elif etype == "subtitle":
                text = str(ev.get("text", "")).strip()
                is_final = bool(ev.get("is_final", False))
                if text:
                    labeled = self.speaker_labeler.apply(ev)
                    speaker_label = str(labeled.get("speaker_label", ""))
                    self._apply_incoming_subtitle(text, is_final, speaker_label)

            elif etype == "heartbeat":
                if self.is_live_asr_active and self.sub_mode == "대기 중":
                    self.sub_status_label.setText("자막: 연결 유지 중")

            elif etype == "asr_error":
                err_msg = str(ev.get("message", "WLK 오류"))
                self.is_live_asr_active = False
                self.live_asr_btn.setText("실시간 자막 시작")
                self._show_toast(err_msg)
                if self.is_recording:
                    self.set_record_only_mode()
                else:
                    self.sub_mode = "대기 중"
                    self.sub_status_label.setText("자막: 연결 오류")

    @staticmethod
    def _normalize_sentence(text: str) -> str:
        return text.lstrip("- ").rstrip(".?!, ").strip()

    def _is_same_utterance_continuation(self, prev_raw: str, new_raw: str) -> bool:
        p = self._normalize_sentence(prev_raw)
        n = self._normalize_sentence(new_raw)
        if not p or not n:
            return False
        # 앞 문장이 새 문장의 접두사거나(확장), 반대로 살짝 줄어든 동일 발화인 경우
        return n.startswith(p) or p.startswith(n)

    def _replace_last_history_line(self, formatted_text: str) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        line = f"[{now_str}] {formatted_text}"
        if not self.history_entries:
            self.append_history_line(formatted_text)
            return

        self.history_entries[-1] = line
        self.history_browser.setPlainText("\n".join(self.history_entries))

        if self.search_input.text().strip():
            self._rebuild_search_spans(self.search_input.text().strip(), keep_current=True)

        vbar = self.history_browser.verticalScrollBar()
        if self.auto_scroll:
            vbar.setValue(vbar.maximum())

    def _apply_incoming_subtitle(
        self, text: str, is_final: bool, speaker_label: str = ""
    ) -> None:
        formatted = f"[{speaker_label}] {text}" if speaker_label else text

        if is_final:
            has_valid_current = (
                self.current_is_final
                and bool(self.current_text)
                and not self.current_text.startswith("WLK 서버로")
            )

            if has_valid_current:
                same_speaker = (self.current_speaker_label == speaker_label)
                if same_speaker and self._is_same_utterance_continuation(
                    self.last_committed_raw, text
                ):
                    # 같은 문장이 계속 자라나는 중이면 이전 자막 박스/기록에 새 줄을 쌓지 않음
                    pass
                else:
                    # 완전히 새로운 문장이 시작됐을 때만 직전 완성 문장을 이전 자막 박스와 기록에 확정 저장
                    self.prev_text = self.current_text
                    self.append_history_line(self.current_text)

            self.last_committed_raw = text
            self.last_committed_speaker = speaker_label

        self.current_text = formatted
        self.current_is_final = is_final
        self.current_speaker_label = speaker_label
        self.sub_mode = "수신 중"
        self.sub_status_label.setText("자막: 실시간 수신 중")
        self.critical_alert_banner.hide()
        self.record_only_timer.stop()
        self.record_only_seconds = 0
        self.record_only_counter_label.setText("중단 경과: 0초 / 30초")
        self.refresh_subtitles()

    def toggle_live_asr(self) -> None:
        if self.is_playing_sample:
            self.toggle_sample_subtitles()

        if not self.is_live_asr_active:
            url = self.wlk_url_input.text().strip() or "ws://127.0.0.1:8000/asr"
            self.rec_cmd_queue.put({"type": "set_asr_streaming", "enabled": True})
            self.asr_cmd_queue.put({"type": "start_stream", "url": url})
            self.sub_status_label.setText("자막: WLK 연결 시도 중...")
        else:
            # 명세서 7번: 일시정지 시 기존 스트림 종료 신호 후 연결 닫기
            self.rec_cmd_queue.put({"type": "set_asr_streaming", "enabled": False})
            self.asr_cmd_queue.put({"type": "pause_stream"})

    def on_device_changed(self, combo_idx: int) -> None:
        dev_idx = self.mic_combo.itemData(combo_idx)
        self.rec_cmd_queue.put({"type": "select_device", "device_index": dev_idx})

    def request_refresh_devices(self) -> None:
        self.rec_cmd_queue.put({"type": "refresh_devices"})

    def toggle_recording(self) -> None:
        if not self.is_recording:
            self.rec_cmd_queue.put({"type": "start_recording"})
        else:
            self.rec_cmd_queue.put({"type": "stop_recording"})

    def closeEvent(self, event) -> None:
        try:
            self.worker_poll_timer.stop()
            if hasattr(self, "rec_cmd_queue"):
                self.rec_cmd_queue.put({"type": "shutdown"})
            if hasattr(self, "asr_cmd_queue"):
                self.asr_cmd_queue.put({"type": "shutdown"})
            for proc in (getattr(self, "recorder_proc", None), getattr(self, "asr_proc", None)):
                if proc is not None and proc.is_alive():
                    proc.join(timeout=2.0)
                    if proc.is_alive():
                        proc.terminate()
        except Exception:
            pass
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        w = self.body_widget.width() if self.body_widget.width() > 0 else self.width()
        self.prev_subtitle_box.apply_width_for_parent(w)
        self.current_subtitle_box.apply_width_for_parent(w)

    def on_history_scroll_changed(self, value: int) -> None:
        vbar = self.history_browser.verticalScrollBar()
        at_bottom = value >= vbar.maximum() - 4
        if at_bottom:
            self.auto_scroll = True
            self.unread_new_count = 0
            self.scroll_paused_badge.hide()
            self.return_latest_btn.hide()
        else:
            self.auto_scroll = False
            self.scroll_paused_badge.show()
            if self.unread_new_count > 0:
                self.return_latest_btn.setText(f"새 자막 {self.unread_new_count}개 · 최신으로")
                self.return_latest_btn.show()

    def jump_to_latest(self) -> None:
        self.auto_scroll = True
        self.unread_new_count = 0
        self.scroll_paused_badge.hide()
        self.return_latest_btn.hide()
        vbar = self.history_browser.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def append_history_line(self, text: str) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        line = f"[{now_str}] {text}"
        self.history_entries.append(line)
        self.history_browser.append(line)

        if self.search_input.text().strip():
            self._rebuild_search_spans(self.search_input.text().strip(), keep_current=True)

        vbar = self.history_browser.verticalScrollBar()
        if self.auto_scroll:
            vbar.setValue(vbar.maximum())
        else:
            self.unread_new_count += 1
            self.scroll_paused_badge.show()
            self.return_latest_btn.setText(f"새 자막 {self.unread_new_count}개 · 최신으로")
            self.return_latest_btn.show()

    def on_search_text_changed(self, text: str) -> None:
        query = text.strip()
        if not query:
            self.search_spans = []
            self.search_cursor_idx = -1
            self.last_search_query = ""
            self.search_count_label.setText("")
            self.history_browser.setExtraSelections([])
            return
        self._rebuild_search_spans(query, keep_current=False)
        if self.search_spans:
            self.search_cursor_idx = 0
            self._apply_search_highlight_and_jump()
        else:
            self.search_cursor_idx = -1
            self.search_count_label.setText("0개")
            self.history_browser.setExtraSelections([])

    def _rebuild_search_spans(self, query: str, keep_current: bool) -> None:
        doc = self.history_browser.document()
        spans: list[tuple[int, int]] = []
        cursor = QTextCursor(doc)
        while True:
            found = doc.find(query, cursor, QTextDocument.FindFlag(0))
            if found.isNull():
                break
            spans.append((found.selectionStart(), found.selectionEnd()))
            cursor = found

        self.search_spans = spans
        self.last_search_query = query
        if not spans:
            self.search_cursor_idx = -1
            self.search_count_label.setText("0개")
        elif not keep_current or self.search_cursor_idx < 0 or self.search_cursor_idx >= len(spans):
            self.search_cursor_idx = 0
            self.search_count_label.setText(f"1 / {len(spans)}")
        else:
            self.search_count_label.setText(f"{self.search_cursor_idx + 1} / {len(spans)}")

    def _apply_search_highlight_and_jump(self) -> None:
        if not self.search_spans or self.search_cursor_idx < 0:
            self.history_browser.setExtraSelections([])
            return

        doc = self.history_browser.document()
        normal_fmt = QTextCharFormat()
        normal_fmt.setBackground(QColor("#854D0E"))
        normal_fmt.setForeground(QColor("#FEF08A"))

        active_fmt = QTextCharFormat()
        active_fmt.setBackground(QColor("#EA580C"))
        active_fmt.setForeground(QColor("#FFFFFF"))

        extra_selections: list[QTextBrowser.ExtraSelection] = []
        for idx, (start, end) in enumerate(self.search_spans):
            cur = QTextCursor(doc)
            cur.setPosition(start)
            cur.setPosition(end, QTextCursor.MoveMode.KeepAnchor)

            sel = QTextBrowser.ExtraSelection()
            sel.cursor = cur
            sel.format = active_fmt if idx == self.search_cursor_idx else normal_fmt
            extra_selections.append(sel)

        self.history_browser.setExtraSelections(extra_selections)
        self.search_count_label.setText(f"{self.search_cursor_idx + 1} / {len(self.search_spans)}")

        self.auto_scroll = False
        self.scroll_paused_badge.show()
        if self.unread_new_count > 0:
            self.return_latest_btn.setText(f"새 자막 {self.unread_new_count}개 · 최신으로")
            self.return_latest_btn.show()

        active_start, active_end = self.search_spans[self.search_cursor_idx]
        target_cursor = QTextCursor(doc)
        target_cursor.setPosition(active_start)
        target_cursor.setPosition(active_end, QTextCursor.MoveMode.KeepAnchor)
        self.history_browser.setTextCursor(target_cursor)
        self.history_browser.ensureCursorVisible()

    def find_next_match(self) -> None:
        query = self.search_input.text().strip()
        if not query:
            return
        if query != self.last_search_query or not self.search_spans:
            self._rebuild_search_spans(query, keep_current=False)
            if not self.search_spans:
                return
            self.search_cursor_idx = 0
        else:
            self.search_cursor_idx = (self.search_cursor_idx + 1) % len(self.search_spans)
        self._apply_search_highlight_and_jump()

    def find_prev_match(self) -> None:
        query = self.search_input.text().strip()
        if not query:
            return
        if query != self.last_search_query or not self.search_spans:
            self._rebuild_search_spans(query, keep_current=False)
            if not self.search_spans:
                return
            self.search_cursor_idx = len(self.search_spans) - 1
        else:
            self.search_cursor_idx = (self.search_cursor_idx - 1) % len(self.search_spans)
        self._apply_search_highlight_and_jump()

    def set_delayed_mode(self) -> None:
        self.sample_timer.stop()
        self.record_only_timer.stop()
        self.record_only_seconds = 0
        self.record_only_counter_label.setText("중단 경과: 0초 / 30초")
        self.critical_alert_banner.hide()
        self.sub_mode = "지연"
        self.sub_status_label.setText("자막: 지연 발생")
        self.refresh_subtitles()

    def set_record_only_mode(self) -> None:
        self.sample_timer.stop()
        self.sub_mode = "중단(녹음 유지)"
        self.sub_status_label.setText("자막: 중단·녹음 유지(RECORD_ONLY)")
        if not self.record_only_timer.isActive():
            self.record_only_timer.start()
        self.refresh_subtitles()

    def trigger_30s_alert_test(self) -> None:
        self.set_record_only_mode()
        self.record_only_seconds = 31
        self.record_only_counter_label.setText(f"중단 경과: {self.record_only_seconds}초 / 30초")
        self.critical_alert_banner.setText(
            f"[경고] 녹음만 유지되고 자막 중단이 {self.record_only_seconds}초 지속됨(30초 허용 목표 초과) — 로컬 전환/상태 확인 필요"
        )
        self.critical_alert_banner.show()

    def restore_normal_mode(self) -> None:
        self.record_only_timer.stop()
        self.record_only_seconds = 0
        self.record_only_counter_label.setText("중단 경과: 0초 / 30초")
        self.critical_alert_banner.hide()
        if self.is_playing_sample:
            self.sub_mode = "수신 중"
            self.sub_status_label.setText("자막: 수신 중(테스트)")
            self.sample_timer.start()
        elif self.is_live_asr_active:
            self.sub_mode = "수신 중"
            self.sub_status_label.setText("자막: 실시간 수신 중")
        else:
            self.sub_mode = "대기 중"
            self.sub_status_label.setText("자막: 대기 중")
        self.refresh_subtitles()

    def tick_record_only(self) -> None:
        self.record_only_seconds += 1
        self.record_only_counter_label.setText(f"중단 경과: {self.record_only_seconds}초 / 30초")
        if self.record_only_seconds > self.alert_threshold_sec:
            self.critical_alert_banner.setText(
                f"[경고] 녹음만 유지되고 자막 중단이 {self.record_only_seconds}초 지속됨(30초 허용 목표 초과) — 로컬 전환/상태 확인 필요"
            )
            self.critical_alert_banner.show()

    def refresh_subtitles(self) -> None:
        self.font_size_label.setText(f"{self.font_size}px")
        prev_size = max(14, int(self.font_size * 0.7))
        w = self.body_widget.width() if self.body_widget.width() > 0 else self.width()

        if self.prev_text:
            self.prev_subtitle_box.show()
            self.prev_subtitle_box.set_subtitle_text(self.prev_text, prev_size, is_final=True)
            self.prev_subtitle_box.apply_width_for_parent(w)
        else:
            self.prev_subtitle_box.hide()

        show_recognizing = (
            (self.is_playing_sample or self.is_live_asr_active)
            and self.sub_mode == "수신 중"
            and not self.current_is_final
        )
        if show_recognizing:
            self.state_badge.setText("인식 중")
            self.state_badge.show()
        else:
            self.state_badge.hide()

        self.current_subtitle_box.set_subtitle_text(
            self.current_text, self.font_size, is_final=self.current_is_final
        )
        self.current_subtitle_box.apply_width_for_parent(w)

    def toggle_focus_mode(self) -> None:
        self.is_focus_mode = not self.is_focus_mode
        if self.is_focus_mode:
            self.mic_bar.hide()
            self.sim_bar.hide()
            self.history_section.hide()
            self.focus_btn.setText("일반 화면으로")
            self.focus_hint_label.show()
        else:
            self.mic_bar.show()
            self.sim_bar.show()
            self.history_section.show()
            self.focus_btn.setText("자막 집중 화면")
            self.focus_hint_label.hide()
        self.refresh_subtitles()

    def exit_focus_mode(self) -> None:
        if self.is_focus_mode:
            self.toggle_focus_mode()

    def increase_font(self) -> None:
        if self.font_size < self.max_font_size:
            self.font_size += 4
            self.refresh_subtitles()

    def decrease_font(self) -> None:
        if self.font_size > self.min_font_size:
            self.font_size -= 4
            self.refresh_subtitles()

    def toggle_sample_subtitles(self) -> None:
        if self.is_live_asr_active:
            self.toggle_live_asr()

        self.is_playing_sample = not self.is_playing_sample
        if self.is_playing_sample:
            self.sample_btn.setText("테스트 자막 정지")
            self.sub_mode = "수신 중"
            self.sub_status_label.setText("자막: 수신 중(테스트)")
            self.critical_alert_banner.hide()
            self.record_only_timer.stop()
            self.push_next_sample()
            self.sample_timer.start()
        else:
            self.sample_btn.setText("테스트 자막 재생")
            self.sub_mode = "대기 중"
            self.sub_status_label.setText("자막: 대기 중")
            self.sample_timer.stop()
            self.current_is_final = True
            self.refresh_subtitles()

    def push_next_sample(self) -> None:
        if (
            self.current_is_final
            and self.current_text
            and not self.current_text.startswith("WLK 서버로")
        ):
            self.prev_text = self.current_text
            self.append_history_line(self.current_text)

        line, is_final = SAMPLE_STREAM[self.sample_index % len(SAMPLE_STREAM)]
        self.sample_index += 1
        self.current_text = line
        self.current_is_final = is_final
        self.refresh_subtitles()

    def add_bookmark(self) -> None:
        now_str = datetime.now().strftime("%H:%M:%S")
        memo = self.current_text[:24] + ("..." if len(self.current_text) > 24 else "")
        self.bookmarks.append(f"[{now_str}] {memo}")
        self.bookmark_info_btn.setText(f"북마크 {len(self.bookmarks)}개")
        self._show_toast(f"[{now_str}] 이 부분 다시 확인 저장됨!")

    def show_bookmarks_dialog(self) -> None:
        if not self.bookmarks:
            QMessageBox.information(self, "북마크 목록", "아직 저장된 북마크가 없습니다.")
            return
        QMessageBox.information(self, "북마크 목록", "\n".join(self.bookmarks))