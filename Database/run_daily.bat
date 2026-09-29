@echo off
rem Daily pipeline run (Windows Task Scheduler "MarketDataPipeline", Mon-Sat 22:00).
rem Output of each run: data\logs\daily_YYYY-MM-DD_HHMM.log
rem Saturdays: weekly copy of raw.duckdb (clinical-trial versions can't be re-downloaded) and
rem corporate_ids.csv to %USERPROFILE%\Backups\database_weekly\, the last 8 copies of each are kept.

cd /d "%~dp0"
if not exist data\logs mkdir data\logs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HHmm"') do set STAMP=%%i
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).DayOfWeek"') do set DOW=%%i
set LOG=data\logs\daily_%STAMP%.log
set PYTHONIOENCODING=utf-8

echo Start %DATE% %TIME% > "%LOG%"
"C:\Program Files\Python313\python.exe" run.py >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo End %DATE% %TIME% (exit code %RC%) >> "%LOG%"

if /i "%DOW%"=="Saturday" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backup_weekly.ps1" >> "%LOG%" 2>&1
)
exit /b %RC%
