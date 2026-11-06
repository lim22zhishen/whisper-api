# Manages the Whisper API as a scheduled task that starts when Windows boots.
# Usage: powershell -ExecutionPolicy Bypass -File deploy\service.ps1 <action>
# install and uninstall need an Administrator PowerShell.
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('check', 'install', 'start', 'stop', 'restart', 'status', 'uninstall')]
    [string]$Action
)

$ErrorActionPreference = 'Stop'
$taskName = 'WhisperAPI'
$root = Split-Path $PSScriptRoot -Parent
$runner = Join-Path $PSScriptRoot 'run_server.ps1'

function Get-Task {
    Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}

function Get-Problems {
    $problems = @()
    if (-not (Test-Path (Join-Path $root '.venv\Scripts\python.exe'))) {
        $problems += 'No .venv found. Follow the setup steps in README.md first.'
    }
    if (-not (Test-Path (Join-Path $root '.env'))) {
        $problems += 'No .env found. Copy .env.example to .env and set WHISPER_API_KEY.'
    }
    return $problems
}

function New-WhisperTask {
    $arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runner`""
    $taskAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -AtStartup
    # S4U runs at boot with no login and no stored password, without admin rights.
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew
    New-ScheduledTask -Action $taskAction -Trigger $trigger -Principal $principal -Settings $settings `
        -Description 'Whisper speech-to-text API'
}

function Stop-Server {
    if (Get-Task) {
        Stop-ScheduledTask -TaskName $taskName
    }
    # Stopping the task can leave child processes running. Stop the restart loop first.
    $filters = @(
        "Name='powershell.exe' AND CommandLine LIKE '%run_server.ps1%'",
        "Name='python.exe' AND CommandLine LIKE '%-m whisper_api%'"
    )
    foreach ($filter in $filters) {
        Get-CimInstance Win32_Process -Filter $filter | ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}

switch ($Action) {
    'check' {
        $problems = Get-Problems
        $null = New-WhisperTask
        if ($problems) {
            $problems | ForEach-Object { Write-Output "Problem: $_" }
            exit 1
        }
        Write-Output 'Ready to install.'
    }
    'install' {
        $problems = Get-Problems
        if ($problems) {
            $problems | ForEach-Object { Write-Output "Problem: $_" }
            exit 1
        }
        Register-ScheduledTask -TaskName $taskName -InputObject (New-WhisperTask) -Force | Out-Null
        Start-ScheduledTask -TaskName $taskName
        Write-Output "Installed and started. Log: $root\logs\server.log"
    }
    'start' {
        Start-ScheduledTask -TaskName $taskName
        Write-Output 'Started.'
    }
    'stop' {
        Stop-Server
        Write-Output 'Stopped.'
    }
    'restart' {
        Stop-Server
        Start-ScheduledTask -TaskName $taskName
        Write-Output 'Restarted.'
    }
    'status' {
        $task = Get-Task
        if ($task) { Write-Output "Task: $($task.State)" } else { Write-Output 'Task: not installed' }
    }
    'uninstall' {
        Stop-Server
        if (Get-Task) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        }
        Write-Output 'Uninstalled.'
    }
}
