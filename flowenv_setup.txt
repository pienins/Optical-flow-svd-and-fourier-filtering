call conda create -n flowenv python=3.10 -y
call conda activate flowenv
set PYTHONNOUSERSITE=1

python -m ensurepip --upgrade
python -m pip install --upgrade pip

python -m pip install numpy==1.26.4

python -m pip install opencv-python-headless==4.5.5.64 --no-deps

python -c "import cv2; print('OpenCV version:', cv2.__version__); print(cv2.__file__)"

python -m pip install imageio pandas matplotlib tifffile scipy tqdm wolframclient

python -m pip install spyder

python -c "import spyder; print(spyder.__version__)"
python -c "import spyder_kernels; print(spyder_kernels.__version__)"

