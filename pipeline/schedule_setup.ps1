# SaayaShift — Windows Task Scheduler setup
# Run this once as Administrator to register the hourly pipeline task.
# Evidence window: Tue 20 Oct 16:00 PKT through Wed 21 Oct 16:00 PKT

$taskName   = "SaayaShift-Hourly-Pipeline"
$scriptPath = "D:\hackathon\pipeline\run_hourly.bat"
$workDir    = "D:\hackathon"

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Create trigger: run every 1 hour, starting now
$trigger = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Hours 1) -Once -At (Get-Date)

# Run as current user
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Action: run the batch file
$action = New-ScheduledTaskAction -Execute $scriptPath -WorkingDirectory $workDir

# Settings: allow run on battery, run even if on AC, do not stop if going on battery
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0 -Minutes 30) `
    -DisallowStartIfOnBatteries $false `
    -StopIfGoingOnBatteries $false `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Trigger $trigger -Principal $principal -Action $action -Settings $settings -Force

Write-Host "Task '$taskName' registered. Running every hour."
Write-Host "To verify: Get-ScheduledTask -TaskName '$taskName'"
Write-Host "To run now: Start-ScheduledTask -TaskName '$taskName'"
Write-Host "To remove: Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
