@echo off
setlocal

REM Name of the conda environment
set "ENV_NAME=hl2ss"

REM Check if the environment already exists
conda env list | findstr /C:"%ENV_NAME%" >nul
if %errorlevel% == 0 (
    echo Environment '%ENV_NAME%' already exists.
) else (
    echo Creating environment '%ENV_NAME%'...
    conda env create --file environment.yaml
)


REM Activate the environment
echo Activating environment '%ENV_NAME%'...
call conda activate %ENV_NAME%


REM Install or upgrade openmim and other packages if not already installed
call :check_installed openmim
if %errorlevel% == 0 (
    echo openmim is already installed.
) else (
    echo Installing/upgrading openmim...
    pip install -U openmim
)

call :check_installed mmengine
if %errorlevel% == 0 (
    echo mmengine is already installed.
) else (
    echo Installing mmengine...
    mim install mmengine
)

REM Check if the version of mmcv is >= 2.0.0
python -c "import mmcv; assert mmcv.__version__ >= '2.0.0' and mmcv.__version__ < '2.2.0'" >nul 2>&1
if %errorlevel% == 0 (
    echo mmcv>=2.0.0 is already installed.
) else (
    echo Installing/upgrading mmcv to version >= 2.0.0,<2.2.0...
    mim install "mmcv>=2.0.0,<2.2.0"
)

call :check_installed mmdet
if %errorlevel% == 0 (
    echo mmdet is already installed.
) else (
    echo Installing/upgrading mmdet...
    mim install mmdet
)

echo Environment setup complete!

REM Function to check if a Python package is installed
:check_installed
python -c "import %1" >nul 2>&1
set RESULT=%errorlevel%

if %RESULT% neq 0 (
    echo Failed to import %1. Error level: %RESULT%
) else (
    echo Successfully imported %1.
)

exit /b %RESULT%

endlocal