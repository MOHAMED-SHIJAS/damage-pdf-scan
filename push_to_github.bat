@echo off
setlocal
cd /d "%~dp0"
set GIT_EXE="%LOCALAPPDATA%\Programs\Git\cmd\git.exe"

echo ======================================================================
echo   Push Vision Agent Studio to GitHub
echo ======================================================================
echo Target Repository: https://github.com/MOHAMED-SHIJAS/Damage-document-scanner.git
echo Branch:            main
echo ======================================================================
echo.
echo Press ENTER to push to this repository, or enter a different URL:
set "DEFAULT_REPO=https://github.com/MOHAMED-SHIJAS/Damage-document-scanner.git"
set /p REPO_URL="Repository URL [%DEFAULT_REPO%]: "
if "%REPO_URL%"=="" set "REPO_URL=%DEFAULT_REPO%"

%GIT_EXE% remote remove origin >nul 2>&1
%GIT_EXE% remote add origin %REPO_URL%

echo.
echo ----------------------------------------------------------------------
echo Pushing project files to %REPO_URL% ...
echo (If a GitHub sign-in prompt appears, choose 'Sign in with your browser')
echo ----------------------------------------------------------------------
echo.

%GIT_EXE% push -u origin main --force

if %ERRORLEVEL% equ 0 (
    echo.
    echo ======================================================================
    echo   [SUCCESS] Code successfully pushed to GitHub!
    echo.
    echo   Next step: Deploy on Render:
    echo   1. Go to https://dashboard.render.com
    echo   2. Click "New +" -^> "Web Service"
    echo   3. Connect repository: MOHAMED-SHIJAS/Damage-document-scanner
    echo   4. Render will auto-detect render.yaml and deploy!
    echo ======================================================================
) else (
    echo.
    echo ======================================================================
    echo   [FAILED] Push could not complete.
    echo   Please check your GitHub permissions or try with a Personal Access Token.
    echo ======================================================================
)

echo.
pause
