param([string]$Python = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $Python = if (Test-Path -LiteralPath $bundledPython) { $bundledPython } else { 'python' }
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv --without-pip .venv
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать Python-окружение.' }
}
& $Python -m pip --python .venv install torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить CPU PyTorch.' }
& $Python -m pip --python .venv install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить Python-зависимости.' }
$bundledNode = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin'
$bundledPnpm = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd'
if (Test-Path -LiteralPath $bundledNode) { $env:PATH = $bundledNode + ';' + $env:PATH }
$projectPnpm = if (Test-Path -LiteralPath $bundledPnpm) { $bundledPnpm } else { 'pnpm' }
Push-Location -LiteralPath 'frontend'
try {
    & $projectPnpm install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить frontend-зависимости. Нужны Node.js 20.19+ и pnpm.' }
    & $projectPnpm run build
    if ($LASTEXITCODE -ne 0) { throw 'Ошибка сборки интерфейса.' }
} finally { Pop-Location }
Write-Host 'Готово. Для запуска: .\start.ps1' -ForegroundColor Green
