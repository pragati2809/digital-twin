# Digital Twin automation (Windows PowerShell)
# Usage: .\scripts\run_all.ps1 -Task [train|test|up|down|replay|logs|all]

param(
    [string]$Task = "all"
)

Set-Location -Path (Join-Path $PSScriptRoot "..")

function Show-Header {
    param([string]$Title)
    Write-Host "==========================================================" -ForegroundColor DarkCyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor DarkCyan
}

if ($Task -eq "train" -or $Task -eq "all") {
    Show-Header "Phase 1: EDA and model training (CWRU)"
    python scripts/eda.py
    python src/train.py
}

if ($Task -eq "test" -or $Task -eq "all") {
    Show-Header "Phase 2: Pytest suite"
    python -m pytest -v
}

if ($Task -eq "up") {
    Show-Header "Launching the full Docker Compose stack"
    docker compose up --build -d
    Write-Host "Stack online:" -ForegroundColor Green
    Write-Host " - Streamlit: http://localhost:8501" -ForegroundColor Yellow
    Write-Host " - Grafana:   http://localhost:3000 (admin / GRAFANA_ADMIN_PASSWORD, default admin)" -ForegroundColor Yellow
    Write-Host " - InfluxDB:  http://localhost:8086 (localhost only)" -ForegroundColor Yellow
}

if ($Task -eq "down") {
    Show-Header "Stopping the stack"
    docker compose down
}

if ($Task -eq "logs") {
    docker compose logs -f --tail=50
}

if ($Task -eq "replay") {
    Show-Header "Replay publisher outside Docker (Ctrl+C to stop)"
    python src/replay_publisher.py --rate 1.0
}
