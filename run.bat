@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
cd code
echo Running script...
python pipeline/main.py
pause