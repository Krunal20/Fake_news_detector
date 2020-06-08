@echo off
set /p change="Do you want to deploy API somewhere other than http://0.0.0.0:12345/?(y/n) : "
IF %change%==y set /p host="Enter the host IP : "
IF %change%==y set /p port="Enter the port number : "
IF DEFINED host (python api.py %host% %port%) ELSE (python api.py)
pause