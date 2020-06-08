@echo off
IF NOT EXIST model1.pkl (python models.py) ELSE (set /p redump="A previously dumped model detected. Do you want to dump again?(y/n) : ")
IF %redump%==y (python models.py)
echo.
echo Continue to start server and deploy API
echo. 
pause
cls
start.bat
