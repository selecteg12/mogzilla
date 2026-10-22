param([ValidateRange(1, 65535)][int]$Port = 8008)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$mogzillaPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$mogzillaModel = Join-Path $PSScriptRoot 'training\checkpoints\holistic'
if (-not (Test-Path -LiteralPath $mogzillaPython)) { throw 'Не найдено окружение .venv. Выполните setup.ps1.' }
if (-not (Test-Path -LiteralPath (Join-Path $mogzillaModel 'personal-visual.pt'))) { throw 'Не найдены веса текущей модели в training/checkpoints/holistic. Проверьте комплект репозитория.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend\dist\index.html'))) { throw 'Сначала соберите интерфейс: pnpm --dir frontend build' }
Write-Host "mogzilla: http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host 'Текущая модель mogzilla. Остановка: Ctrl+C.'
& $mogzillaPython -X utf8 -m scripts.serve_personal_preview --model-dir $mogzillaModel --port $Port
