@echo off
set PYTHON_BIN=%LOCALAPPDATA%\Programs\Python\Python311\python.exe

if "%1"=="" (
    "%PYTHON_BIN%" main.py
    exit /b %errorlevel%
)

if "%1"=="1" (
    "%PYTHON_BIN%" agent\run_part1.py
) else if "%1"=="2" (
    "%PYTHON_BIN%" agent\run_part2.py
) else if "%1"=="3" (
    "%PYTHON_BIN%" agent\run_part3.py
) else if "%1"=="4" (
    "%PYTHON_BIN%" agent\run_part4.py
) else if "%1"=="5" (
    "%PYTHON_BIN%" agent\run_part5.py
) else if "%1"=="6" (
    "%PYTHON_BIN%" agent\run_part6.py
) else if "%1"=="7" (
    "%PYTHON_BIN%" agent\run_part7.py
) else if "%1"=="8" (
    "%PYTHON_BIN%" agent\run_part8.py
) else if "%1"=="bonus" (
    "%PYTHON_BIN%" agent\run_part8.py
) else if "%1"=="web" (
    "%PYTHON_BIN%" -m streamlit run app.py
) else if "%1"=="ui" (
    "%PYTHON_BIN%" -m streamlit run app.py
) else if "%1"=="app" (
    "%PYTHON_BIN%" -m streamlit run app.py
) else (
    "%PYTHON_BIN%" %*
)

