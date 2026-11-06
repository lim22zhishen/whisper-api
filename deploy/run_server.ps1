# Runs the API and restarts it if it stops. Task Scheduler starts this at boot.
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
New-Item -ItemType Directory -Force logs | Out-Null

while ($true) {
    # cmd does the redirect because PowerShell 5.1 mangles native stderr.
    & cmd /c ".venv\Scripts\python.exe -m whisper_api >> logs\server.log 2>&1"
    Add-Content logs\server.log "$(Get-Date -Format s) Server stopped (exit $LASTEXITCODE), restarting in 30 seconds"
    Start-Sleep -Seconds 30
}
