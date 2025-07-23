@echo off
title AutoCropperPro Launcher & Builder

REM ── Change to your project directory
cd /d "C:\Users\Benjamin\Desktop\Python Apps\Auto-Cropper\auto_cropper_v2"

REM ── 1) Ensure PyInstaller is installed in Python 3.12
py -3.12 -m pip install pyinstaller

REM ── 2) Build the standalone exe (one‑file, windowed, with icon + font)
py -3.12 -m PyInstaller --onefile --windowed --name AutoCropperPro_v2 ^
  --icon app_icon.ico ^
  --add-data "GothamMedium.otf;." ^
  auto_cropper_v2.py

REM ── 3) (Optional) Launch the freshly built exe
if exist dist\AutoCropperPro_v2.exe (
  echo Launching AutoCropperPro_v2...
  start "" dist\AutoCropperPro_v2.exe
) else (
  echo Build failed or dist\AutoCropperPro_v2.exe not found.
)

exit /b
