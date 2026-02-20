@echo off
chcp 65001 >nul
python dev_tools/recalculate_all_rewards.py
pause
