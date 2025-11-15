@echo off
cd /d "%~dp0"
echo Running Visual Product Matcher setup...
where conda >nul 2>&1
if %ERRORLEVEL%==0 (
    call conda create -n vpm python=3.11 -y
    call conda activate vpm
    conda install -c conda-forge scikit-learn numpy pandas -y
    conda install -c pytorch cpuonly -c conda-forge pytorch -y
    pip install transformers ftfy tqdm pillow requests streamlit
) else (
    python -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
    python -m pip install transformers ftfy tqdm pillow scikit-learn pandas numpy requests streamlit
)
python prepare_features_clip.py
streamlit run app.py
pause
