@echo off
echo ==================================================
echo      MedGemma-PD: Clinical Reasoning Demo
echo ==================================================
cd /d "%~dp0"
echo.
echo [1/2] Running Clinical Pipeline (Patient P07)...
python main.py --patient P07 --session 6

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Pipeline failed. Please check the errors above.
    pause
    exit /b
)

echo.
echo [2/2] Opening Clinical Dashboard...
start medgemma_pd\ui\index.html

echo.
echo Done! The dashboard should be open in your browser.
pause
