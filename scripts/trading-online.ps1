# Run from any directory; all relative project paths resolve inside this workspace.
# Examples: .\scripts\trading-online.ps1 run --once
#           .\scripts\trading-online.ps1 status
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw 'Falta .venv\Scripts\python.exe. Crear el entorno desde la raíz del proyecto.'
}
Push-Location -LiteralPath $projectDirectory
try {
    & $pythonExecutable -X utf8 -m trading_ml online @args
    $commandExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $commandExitCode
