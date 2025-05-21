@echo off
echo Starting Healthcare App...

:: Change to the application directory
cd /d "%~dp0"
if not exist "app.py" (
    cd /d "directory to healthcare app"
    if not exist "app.py" (
        echo ERROR: Cannot find app.py
        pause
        exit /b 1
    )
)

:: Verify virtual environment exists
if not exist "venv\Scripts\streamlit.exe" (
    echo ERROR: Streamlit not found in virtual environment.
    echo Installing required packages...
    if not exist "venv\Scripts\activate.bat" (
        echo Creating virtual environment...
        python -m venv venv
    )
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    echo Installation complete.
)

:: Add venv\Scripts to PATH
set "PATH=%PATH%;%CD%\venv\Scripts"

:: Run the application
echo Starting Streamlit application...
call venv\Scripts\activate.bat
venv\Scripts\streamlit.exe run app.py

pause
