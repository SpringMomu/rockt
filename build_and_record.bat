@echo off
rem One-click: rebuild the native Clarabel DLL, run the landing regression,
rem and record landing videos with the game's own renderer.
rem Everything is written to build_and_record.log next to this file.
chcp 65001 >nul
cd /d "%~dp0"
set "LOG=%~dp0build_and_record.log"
set "PYTHONIOENCODING=utf-8"
echo ==== %DATE% %TIME% ==== > "%LOG%"

echo ==== where python / cargo ==== >> "%LOG%"
where python >> "%LOG%" 2>&1
where cargo >> "%LOG%" 2>&1
python --version >> "%LOG%" 2>&1

echo ==== build native solver ==== >> "%LOG%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0native\build_solver.ps1" >> "%LOG%" 2>&1
echo build exit code %ERRORLEVEL% >> "%LOG%"

echo ==== cargo test ==== >> "%LOG%"
cargo test --manifest-path "%~dp0native\Cargo.toml" >> "%LOG%" 2>&1
echo cargo test exit code %ERRORLEVEL% >> "%LOG%"

echo ==== pip (video encoder) ==== >> "%LOG%"
python -m pip install -q imageio-ffmpeg >> "%LOG%" 2>&1

echo ==== landing test ==== >> "%LOG%"
set "SDL_VIDEODRIVER=dummy"
python main.py --landing-test >> "%LOG%" 2>&1
echo landing test exit code %ERRORLEVEL% >> "%LOG%"

echo ==== record videos ==== >> "%LOG%"
python main.py --record videos >> "%LOG%" 2>&1
echo record exit code %ERRORLEVEL% >> "%LOG%"

echo ==== DONE ==== >> "%LOG%"
