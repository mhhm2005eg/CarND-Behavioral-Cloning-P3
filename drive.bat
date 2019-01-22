set Model_name=VN_model
set PYTHON="C:\Users\mhhm2\AppData\Local\Programs\Python\Python36\python.exe"
set out_folder="run1"
start ..\beta_simulator_windows\beta_simulator.exe
%PYTHON% .\drive.py %Model_name% %out_folder%
