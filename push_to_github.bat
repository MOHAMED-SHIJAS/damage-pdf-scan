@echo off
setlocal
cd /d "%~dp0"
set GIT_EXE="%LOCALAPPDATA%\Programs\Git\cmd\git.exe"

echo ======================================================================
echo   Push Vision Agent Studio to GitHub
echo ======================================================================
echo.

set /p REPO_URL="Enter your GitHub repository URL (e.g. https://github.com/username/repo.git): "
if "%REPO_URL%"=="" (
    echo Error: Repository URL cannot be empty.
    pause
    exit /b 1
)

%GIT_EXE% remote remove origin >nul 2>&1
%GIT_EXE% remote add origin %REPO_URL%
echo.
echo Pushing branch 'main' to %REPO_URL%...
%GIT_EXE% push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ======================================================================
    echo   SUCCESS: Pushed to GitHub!
    echo   Now open https://dashboard.render.com and connect this repo.
    echo ======================================================================
) else (
    echo.
    echo Push failed. Check your GitHub URL and credentials/login.
)

pause
