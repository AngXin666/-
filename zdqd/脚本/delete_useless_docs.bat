@echo off
chcp 65001 >nul
echo ========================================
echo 删除无用文档
echo ========================================
echo.
echo 将删除 56 个无用文档：
echo   - 12 个临时修复记录
echo   - 35 个任务总结
echo   - 4 个临时分析报告
echo   - 2 个Kiro临时spec目录
echo   - 3 个其他临时文件
echo.
echo 保留 23 个有用文档：
echo   - 核心文档（README、更新日志）
echo   - 用户指南（加密、启动脚本）
echo   - 开发文档（测试、配置、最佳实践）
echo   - 训练指南（模型训练）
echo.
pause
echo.

REM 临时修复记录
echo [1/6] 删除临时修复记录...
del /f /q "AUTOMATED_TESTING_SUMMARY.md" 2>nul
del /f /q "AUTO_TRANSFER_AUDIT_REPORT.md" 2>nul
del /f /q "BUGFIX_COMPLETE_SUMMARY.md" 2>nul
del /f /q "BUGFIX_IMPORT_ERRORS.md" 2>nul
del /f /q "FIXED_COORDINATE_AUDIT.md" 2>nul
del /f /q "NAVIGATION_UNIFICATION_FIX.md" 2>nul
del /f /q "PROFILE_AD_CLOSE_FIX.md" 2>nul
del /f /q "PROFILE_AD_DETECTION_FIX.md" 2>nul
del /f /q "PROFILE_AD_FIX_AUDIT.md" 2>nul
del /f /q "STARTUP_FLOW_FIX.md" 2>nul
del /f /q "TIME_FORMAT_FIX_SUMMARY.md" 2>nul
del /f /q "TRANSFER_SAFETY_FIX_SUMMARY.md" 2>nul
del /f /q "PROJECT_HEALTH_CHECK.md" 2>nul
echo   完成！

REM 临时分析报告
echo [2/6] 删除临时分析报告...
del /f /q "PROFILE_AD_DETECTION_ANALYSIS.md" 2>nul
del /f /q "checkin_performance_analysis.md" 2>nul
del /f /q "CHECKIN_POPUP_DETECTION_ANALYSIS.md" 2>nul
del /f /q "checkpoint_test_report.md" 2>nul
del /f /q "optimize_checkin_popup_detection.md" 2>nul
echo   完成！

REM 任务总结
echo [3/6] 删除任务总结...
del /f /q "BATCH_ADD_ACCOUNTS_AUTO_REFRESH_SUMMARY.md" 2>nul
del /f /q "BATCH_ADD_ACCOUNTS_FEATURE.md" 2>nul
del /f /q "CODE_CLEANUP_PLAN.md" 2>nul
del /f /q "CODE_CLEANUP_SUMMARY.md" 2>nul
del /f /q "COMPLETE_CLEANUP_SUMMARY.md" 2>nul
del /f /q "ENCRYPTION_IMPLEMENTATION_SUMMARY.md" 2>nul
del /f /q "ERROR_LOG_IMPROVEMENT_SUMMARY.md" 2>nul
del /f /q "FILE_CLEANUP_PLAN.md" 2>nul
del /f /q "FILE_CLEANUP_SUMMARY.md" 2>nul
del /f /q "GPU加速整合方案总结.md" 2>nul
del /f /q "GUI_CLEANUP_SUMMARY.md" 2>nul
del /f /q "GUI_COLUMN_ORDER_FIX_SUMMARY.md" 2>nul
del /f /q "MODEL_SINGLETON_OPTIMIZATION_FINAL_SUMMARY.md" 2>nul
del /f /q "PAGE_CLASSIFIER_INTEGRATION_SUMMARY.md" 2>nul
del /f /q "PAUSE_FUNCTION_FIX_SUMMARY.md" 2>nul
del /f /q "PyTorch模型迁移总结.md" 2>nul
del /f /q "TASK_1_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_2_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_3_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_3_HISTORY_OPTIMIZATION_SUMMARY.md" 2>nul
del /f /q "TASK_4_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_5_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_6_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_7_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_8_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_9_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_10_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_11_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_11.3_COMPLETION_SUMMARY.md" 2>nul
del /f /q "TASK_13_FINAL_VERIFICATION_SUMMARY.md" 2>nul
del /f /q "TRANSFER_INTEGRATED_DETECTOR_SUMMARY.md" 2>nul
del /f /q "TRANSFER_YOLO_OPTIMIZATION.md" 2>nul
del /f /q "docs\task_6_completion_summary.md" 2>nul
del /f /q "docs\task_7_completion_summary.md" 2>nul
del /f /q "docs\error_handling_refactoring_guide.md" 2>nul
echo   完成！

REM Kiro临时spec
echo [4/6] 删除Kiro临时spec目录...
if exist ".kiro\specs\code-quality-improvement" rmdir /s /q ".kiro\specs\code-quality-improvement" 2>nul
if exist ".kiro\specs\multi-recipient-transfer" rmdir /s /q ".kiro\specs\multi-recipient-transfer" 2>nul
echo   完成！

REM 其他临时文件
echo [5/6] 删除其他临时文件...
del /f /q "reports\code_quality_final_report.md" 2>nul
echo   完成！

REM 清理分析脚本
echo [6/6] 清理分析脚本...
del /f /q "analyze_docs.py" 2>nul
del /f /q "review_all_docs.py" 2>nul
del /f /q "classify_docs.py" 2>nul
del /f /q "delete_invalid_docs.bat" 2>nul
del /f /q "review_output.txt" 2>nul
echo   完成！

echo.
echo ========================================
echo 删除完成！
echo ========================================
echo.
echo 已删除 56 个无用文档
echo 保留 23 个有用文档
echo.
pause
