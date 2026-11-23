param([string]$PythonExe = 'python')
Set-Location -LiteralPath $PSScriptRoot
& $PythonExe -m uvicorn app.main:app --app-dir . --host 127.0.0.1 --port 8000
