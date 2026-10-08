# Digital Twin PowerShell Automation Script
# Usage: .\scripts\run_all.ps1 -Task [train|test|up|down|replay]

param(
    [string]$Task = "all"
)

Set-Location -Path $PSScriptRoot\..

function Show-Header {
    param([string]$Title)
    Write-Host "==========================================================" -ForegroundColor Cipher
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cipher
}

if ($Task -eq "train" -or $Task -eq "all") {
    Show-Header "Phase 1: Running EDA & Model Training"
    python scripts/eda.py
    python src/train.py
}

if ($Task -eq "test" -or $Task -eq "all") {
    Show-Header "Phase 4: Running Pytest Suite"
    pytest -v
}

if ($Task -eq "up") {
    Show-Header "Phase 4: Launching Full Docker Compose Stack"
    docker compose up --build -d
    Write-Host "Stack online! Services running:" -ForegroundColor Green
    Write-Host " - Streamlit Dashboard: http://localhost:8501" -ForegroundColor Yellow
    Write-Host " - Grafana Dashboard:   http://localhost:3000 (admin / admin)" -ForegroundColor Yellow
    Write-Host " - InfluxDB Admin UI:   http://localhost:8086" -ForegroundColor Yellow
}

if ($Task -eq "down") {
    Show-Header "Stopping Docker Compose Stack"
    docker compose down
}

if ($Task -eq "replay") {
    Show-Header "Streaming Replay Publisher (Ctrl+C to stop)"
    python src/replay_publisher.py --rate 1.0 --loop
}
