@echo off
setlocal

:: ------------------------------------------------------------
:: CREATE clean environment: fieldenv
:: Supports imports:
::   import os
::   import glob
::   import pickle
::   import numpy as np
::   import pandas as pd
::   import matplotlib.pyplot as plt
::   from tifffile import imread
::   from scipy.ndimage import zoom
:: ------------------------------------------------------------

set ENV_NAME=fieldenv
set PYTHON_VERSION=3.10

echo.
echo This script will create/recreate the Conda environment: %ENV_NAME%
echo Python version: %PYTHON_VERSION%
echo.

call conda --version
if errorlevel 1 goto :conda_missing

echo.
echo Removing existing %ENV_NAME% environment if present...
call conda remove -n %ENV_NAME% --all -y >nul 2>nul

echo.
echo Creating clean environment...
call conda create -n %ENV_NAME% python=%PYTHON_VERSION% pip -y
if errorlevel 1 goto :fail

echo.
echo Activating environment...
call conda activate %ENV_NAME%
if errorlevel 1 goto :fail

:: Prevent Windows user-site packages from leaking into this environment.
:: This was important on your system for previous environments.
set PYTHONNOUSERSITE=1

echo.
echo Verifying interpreter and user-site isolation...
python -c "import sys, site; print('Python:', sys.executable); print('User site enabled:', site.ENABLE_USER_SITE); print('User site:', site.getusersitepackages())"
if errorlevel 1 goto :fail

echo.
echo Upgrading pip tooling...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo.
echo Installing field-processing dependencies...
python -m pip install numpy pandas matplotlib tifffile scipy
if errorlevel 1 goto :fail

echo.
echo Verifying expected imports...
python -c "import os, glob, pickle; import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; from tifffile import imread; from scipy.ndimage import zoom; import scipy, matplotlib, tifffile; print('All expected imports OK'); print('numpy:', np.__version__); print('pandas:', pd.__version__); print('matplotlib:', matplotlib.__version__); print('tifffile:', tifffile.__version__); print('scipy:', scipy.__version__)"
if errorlevel 1 goto :fail

echo.
echo Installing Spyder...
python -m pip install spyder
if errorlevel 1 goto :fail

echo.
echo Verifying Spyder...
python -c "import spyder; print('Spyder:', spyder.__version__)"
python -c "import spyder_kernels; print('spyder-kernels:', spyder_kernels.__version__)"
if errorlevel 1 goto :fail

echo.
echo Setup complete.
echo.
echo To use this environment later:
echo     conda activate %ENV_NAME%
echo     set PYTHONNOUSERSITE=1
echo.
echo If using Spyder from another environment, set the interpreter to:
python -c "import sys; print('    ' + sys.executable)"
echo.

goto :end

:conda_missing
echo.
echo ERROR: conda was not found. Run this script from Anaconda Prompt.
goto :end

:fail
echo.
echo ERROR: Setup failed. Scroll up to find the first error message.
echo You can remove the partial env with:
echo     conda deactivate
echo     conda remove -n %ENV_NAME% --all -y
goto :end

:end
endlocal
pause
