# Compatibility entry point for the original build command.
from pathlib import Path
spec = Path(SPECPATH) / 'build.spec'
exec(compile(spec.read_text(encoding='utf-8'), str(spec), 'exec'))
