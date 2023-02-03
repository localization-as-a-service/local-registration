@echo off

set experiment=%1
set trial=%2
set subject=%3
set sequence=%4

if "%experiment%"=="" (echo "Experiment name is required" & exit /b 1)
if "%trial%"=="" (echo "Trial name is required" & exit /b 1)
if "%subject%"=="" (echo "Subject name is required" & exit /b 1)
if "%sequence%"=="" (echo "Sequence name is required" & exit /b 1)

if exist "data\raw_data\exp_%experiment%\trial_%trial%\secondary\subject-%subject%\0%sequence%" (echo "Sequence already exists" & exit /b 1)

start "IMU" cmd /c "conda activate local-reg & python lidar_imu_capture.py --experiment %experiment% --trial %trial% --subject %subject% --sequence %sequence% --mode imu"
start "Depth" cmd /c "conda activate local-reg & python lidar_imu_capture.py --experiment %experiment% --trial %trial% --subject %subject% --sequence %sequence% --mode cam"