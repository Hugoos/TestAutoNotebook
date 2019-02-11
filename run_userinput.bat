@echo off
pushd %cd%
:start
set /p "id=Please enter the notebook id to convert: "
call python auto-jupyter-notebook.py %id%
goto start
pause