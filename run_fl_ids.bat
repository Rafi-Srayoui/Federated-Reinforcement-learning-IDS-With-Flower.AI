:: run_rl_fl_ids.bat  –  launch Flower server + N clients
:: ------------------------------------------------------
:: 1. Edit NUM_CLIENTS if you change it in client.py / server.py
:: 2. Double‑click or run from PowerShell/cmd:  run_rl_fl_ids.bat
:: ------------------------------------------------------
REM ------------- user settings ---------------------------------
set NUM_CLIENTS=4
set PY=python      
REM -------------------------------------------------------------

setlocal EnableDelayedExpansion
set /a LAST=NUM_CLIENTS-1

echo Launching Flower server...
start "" cmd /k %PY% rl_fl_server.py
timeout /t 5 > nul

echo Launching %NUM_CLIENTS% clients...
for /l %%i in (0,1,!LAST!) do (
    start "" cmd /k %PY% rl_fl_client.py %%i
    timeout /t 1 > nul
)

echo All processes started.
endlocal


