import sys
from multiprocessing import freeze_support
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from wlk_server_manager import WlkServerManager


def main() -> None:
    freeze_support()
    app = QApplication(sys.argv)
    app.setFont(QFont("Malgun Gothic", 10))

    server_manager = WlkServerManager(
        model="tiny",
        backend="faster-whisper",
        language="ko",
        host="127.0.0.1",
        port=8000,
    )
    started = server_manager.start_if_needed()
    print(f"[APP] start_if_needed() -> {started}", file=sys.stderr, flush=True)
    app.aboutToQuit.connect(server_manager.stop_if_started)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()