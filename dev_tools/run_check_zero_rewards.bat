@echo off
chcp 65001 >nul
python dev_tools/check_zero_rewards_detail.py
pause
