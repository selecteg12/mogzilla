param([int]$Port = 8001)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Не найдено окружение .venv. Выполните setup.ps1.'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend\dist\index.html'))) {
    throw 'Сначала соберите интерфейс: cd frontend; pnpm install; pnpm build'
}
Write-Host "mogzilla: http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host 'Остановка: Ctrl+C. Фото обрабатываются локально в памяти.'
& $projectPython -X utf8 -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port --no-access-log
