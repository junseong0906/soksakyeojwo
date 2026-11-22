import os
import sys
from multiprocessing import freeze_support
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from engine_controller import EngineController
from model_catalog import get_model
from networking import configure_network_tls

def main():
    freeze_support()
    configure_network_tls()
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')
    app = QApplication(sys.argv)
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen' and os.name == 'nt':
        QFontDatabase.addApplicationFont(os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', 'malgun.ttf'))
    app.setFont(QFont('Malgun Gothic', 10))
    controller = EngineController()
    smoke = '--smoke-test' in sys.argv
    demo = '--demo' in sys.argv
    window = MainWindow(controller, start_workers=not (demo or smoke))
    window.show()
    # Restore only complete local installations; first launch never downloads large models.
    model = get_model(controller.settings.value['model_id'])
    diar = bool(controller.settings.value['diarization'])
    device = controller.settings.value['device']
    if not (demo or smoke) and controller.models.installed(model) and controller.runtimes.ready(model, diar, device):
        QTimer.singleShot(150, lambda: controller.apply(model.id, diar, device))
    if demo: QTimer.singleShot(200, window.toggle_sample_subtitles)
    if smoke:
        def verify_package():
            from pathlib import Path
            from multiprocessing import Process, Queue
            from workers import runtime_probe
            from app_paths import atomic_json
            report = Path(sys.argv[sys.argv.index('--smoke-test') + 1])
            result = {}
            try:
                from huggingface_hub import HfApi, hf_hub_url
                import soxr
                events = Queue()
                child = Process(target=runtime_probe, args=(events,))
                child.start()
                result['worker'] = events.get(timeout=10)
                child.join(timeout=3)
                if child.is_alive(): child.terminate()
                window._apply_incoming_subtitle('실시간 자막과 화자 분리를 확인합니다.', True, '교수님')
                window.open_model_settings()
                app.processEvents()
                result['models'] = window.model_dialog.table.rowCount()
                result['history'] = window.history_entries
                result['download_client'] = HfApi.__name__
                import ssl
                import platform
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                result['python'] = platform.python_version()
                result['tls_verification'] = context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
                result['ok'] = result['worker'] == 'worker-ready' and result['models'] == 7
                window.model_dialog.grab().save(str(report.with_suffix('.png')))
                if '--download-smoke' in sys.argv:
                    import traceback
                    def finish_download():
                        if controller.busy: return
                        result['download_ok'] = controller.models.installed(get_model('whisper-tiny'))
                        result['download_status'] = controller.status
                        result['ok'] = result['ok'] and result['download_ok']
                        window.model_dialog.grab().save(str(report.with_suffix('.png')))
                        atomic_json(report, result)
                        app.quit()
                    controller.changed.connect(finish_download)
                    window.model_dialog.download_buttons[0].click()
                    return
            except Exception as exc:
                result['error'] = repr(exc)
            atomic_json(report, result)
            app.quit()
        QTimer.singleShot(250, verify_package)
    result = app.exec()
    controller.shutdown()
    return result

if __name__ == '__main__':
    sys.exit(main())
