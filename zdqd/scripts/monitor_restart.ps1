# Monitor program startup to detect infinite restart
$exePath = "dist\自动签到助手_v1.8.8.exe"
$monitorDuration = 30
$checkInterval = 1

Write-Host "Start monitoring..." -ForegroundColor Green
Write-Host "Duration: $monitorDuration seconds" -ForegroundColor Yellow
Write-Host ""

# Start program
Write-Host "Starting: $exePath" -ForegroundColor Cyan
Start-Process -FilePath $exePath

# Monitor process creation
$processCount = 0
$processHistory = @()
$startTime = Get-Date

for ($i = 0; $i -lt $monitorDuration; $i++) {
    Start-Sleep -Seconds $checkInterval
    
    $processes = Get-Process -ErrorAction SilentlyContinue | Where-Object {$_.ProcessName -like "*AutoSignHelper*" -or $_.MainWindowTitle -like "*签到助手*"}
    
    $currentCount = $processes.Count
    $elapsed = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 1)
    
    if ($currentCount -ne $processCount) {
        $change = if ($currentCount -gt $processCount) { "INCREASE" } else { "DECREASE" }
        Write-Host "[$elapsed s] Process count $change : $processCount -> $currentCount" -ForegroundColor $(if ($currentCount -gt $processCount) { "Red" } else { "Green" })
        
        if ($processes) {
            $processes | ForEach-Object {
                Write-Host "  - PID: $($_.Id), Name: $($_.ProcessName), CPU: $($_.CPU)" -ForegroundColor Gray
            }
        }
        
        $processCount = $currentCount
    }
    
    $processHistory += [PSCustomObject]@{
        Time = $elapsed
        Count = $currentCount
    }
    
    if ($currentCount -gt 3) {
        Write-Host ""
        Write-Host "WARNING: Multiple processes detected ($currentCount), possible infinite restart!" -ForegroundColor Red
        Write-Host ""
        break
    }
}

Write-Host ""
Write-Host "Monitoring complete!" -ForegroundColor Green
Write-Host ""

$maxCount = ($processHistory | Measure-Object -Property Count -Maximum).Maximum
$avgCount = [math]::Round(($processHistory | Measure-Object -Property Count -Average).Average, 2)

Write-Host "Statistics:" -ForegroundColor Cyan
Write-Host "  Max processes: $maxCount" -ForegroundColor $(if ($maxCount -gt 2) { "Red" } else { "Green" })
Write-Host "  Avg processes: $avgCount" -ForegroundColor $(if ($avgCount -gt 1.5) { "Yellow" } else { "Green" })
Write-Host ""

if ($maxCount -le 2) {
    Write-Host "PASS: Program started normally, no infinite restart" -ForegroundColor Green
} elseif ($maxCount -le 5) {
    Write-Host "WARNING: Multiple processes detected" -ForegroundColor Yellow
} else {
    Write-Host "FAIL: Infinite restart detected" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to close all processes..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Get-Process -ErrorAction SilentlyContinue | Where-Object {$_.ProcessName -like "*AutoSignHelper*" -or $_.MainWindowTitle -like "*签到助手*"} | Stop-Process -Force
Write-Host "All processes closed" -ForegroundColor Green
