@echo off
REM SaayaShift hourly pipeline runner
REM Called by Windows Task Scheduler every hour
cd /d D:\hackathon
python pipeline\flows\hourly_flow.py >> pipeline\scheduler_log.txt 2>&1
