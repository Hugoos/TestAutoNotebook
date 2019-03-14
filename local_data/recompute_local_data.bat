@echo off
pushd %cd%

echo "Running this code will recompute all local_data needed for Auto-Jupyter-Notebook."
echo "This is required in order to compute dataset similarity for datasets that were added after running this code last."
echo "All current data within the local_data folder will be lost, in case of error please use the Backup folder to restore the previous local data."
:choice
set /P c=Are you sure you want to continue[Y/N]?
if /I "%c%" EQU "Y" goto :run_code
if /I "%c%" EQU "N" goto :quit
goto :choice


:run_code

echo "Local data will be recomputed, please do not exit this window until done."
python -c "import localDataOpenMLInterface;localDataOpenMLInterface.generateLocalData()" 
pause 
exit

:quit

echo "Aborted operation."
pause 
exit


