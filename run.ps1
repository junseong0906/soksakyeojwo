$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$appPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $appPython)) {
    Write-Host 'Python 3.12와 인터넷 연결이 필요합니다. 실행 파일 배포본은 Python을 별도로 설치하지 않아도 됩니다.'
    & py -3.12 -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12를 설치한 뒤 다시 실행해 주세요.' }
    & $appPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw '앱 의존성 설치 실패. 인터넷 연결을 확인해 주세요.' }
}
& $appPython (Join-Path $PSScriptRoot 'app.py')
