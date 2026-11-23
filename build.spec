# Build with: python -m PyInstaller --noconfirm --distpath ../windows build.spec
from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path
import sys

# Build artifacts must use an unmodified Python distribution. Host-specific SSL
# policies depend on host-only files and cannot be shipped as portable stdlib.
import ssl
ssl_source = Path(ssl.__file__)
if ssl_source.is_file() and 'codex-rejected-cas.pem' in ssl_source.read_text(encoding='utf-8'):
    raise RuntimeError('표준 Python 3.12 환경에서 빌드해 주세요. 호스트 전용 SSL 런타임은 배포할 수 없습니다.')

a = Analysis(
    ['app.py'], pathex=[], binaries=[], datas=[],
    hiddenimports=collect_submodules('huggingface_hub'),
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtQml', 'PySide6.QtQuick'],
    noarchive=False, optimize=0,
)
# Select OpenSSL from the same interpreter instead of unrelated tools on PATH.
for filename in ('libcrypto-3.dll', 'libssl-3.dll', 'libcrypto-3-x64.dll', 'libssl-3-x64.dll'):
    candidates = [Path(sys.base_prefix) / 'DLLs' / filename, Path(sys.base_prefix) / filename]
    matching = next((path for path in candidates if path.is_file()), None)
    if matching:
        a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() != filename.lower()]
        a.binaries.append((filename, str(matching), 'BINARY'))
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='속삭여줘',
          debug=False, strip=False, upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='속삭여줘')
