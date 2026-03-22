@echo off
REM ============================================================
REM setup_from_basic.bat
REM Copies unchanged files from BASIC into HY-BAS-ADV1
REM Run this ONCE from the HY-BAS-ADV1 directory
REM ============================================================

set BASIC=..\BASIC

echo Copying unchanged files from BASIC...

REM -- Models (unchanged) --
copy "%BASIC%\models\backbone.py" "models\backbone.py"
copy "%BASIC%\models\reconstructor.py" "models\reconstructor.py"

REM -- Data directory --
if not exist "data" mkdir data
copy "%BASIC%\data\dataset.py" "data\dataset.py"
copy "%BASIC%\data\collate.py" "data\collate.py"
if exist "%BASIC%\data\__init__.py" copy "%BASIC%\data\__init__.py" "data\__init__.py"

REM -- Losses directory --
if not exist "losses" mkdir losses
copy "%BASIC%\losses\distance_loss.py" "losses\distance_loss.py"
copy "%BASIC%\losses\constraint_loss.py" "losses\constraint_loss.py"
if exist "%BASIC%\losses\__init__.py" copy "%BASIC%\losses\__init__.py" "losses\__init__.py"

REM -- Utils directory --
if not exist "utils" mkdir utils
copy "%BASIC%\utils\submission.py" "utils\submission.py"
if exist "%BASIC%\utils\__init__.py" copy "%BASIC%\utils\__init__.py" "utils\__init__.py"

echo.
echo Done. Now you can run:
echo   python train_adv1.py --config config_adv1.yaml --resume %BASIC%\checkpoints\best_model.pt
echo.
