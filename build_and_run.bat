@echo off
title AutoCropperPro Launcher & Builder

REM ── Change to your project directory
cd /d "C:\Users\User\OneDrive\Desktop\AutoCropperPro\auto-cropper-pro"

REM ── 1) Ensure PyInstaller is installed in Python 3.12
py -3.12 -m pip install pyinstaller

REM ── 2) Build the standalone exe (one‑file, windowed, with icon + font)
py -3.12 -m PyInstaller --onefile --windowed --name AutoCropperPro ^
  --icon app_icon.ico ^
  --add-data "GothamMedium.otf;." ^
  auto_cropper.py

REM ── 3) (Optional) Launch the freshly built exe
if exist dist\AutoCropperPro.exe (
  echo Launching AutoCropperPro...
  start "" dist\AutoCropperPro.exe
) else (
  echo Build failed or dist\AutoCropperPro.exe not found.
)

exit /b
