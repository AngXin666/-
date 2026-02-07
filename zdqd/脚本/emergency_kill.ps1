# 紧急结束所有进程
Write-Host "紧急结束所有进程..." -ForegroundColor Red

# 结束所有相关进程
Get-Process | Where-Object {
    $_.ProcessName -like "*自动签到助手*" -or
    $_.ProcessName -eq "wmic" -or
    $_.ProcessName -eq "mshta" -or
    $_.ProcessName -eq "python" -or
    $_.ProcessName -eq "pythonw"
} | Stop-Process -Force

Write-Host "完成！" -ForegroundColor Green
