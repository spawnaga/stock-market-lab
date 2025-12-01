@echo off
REM Distributed Training Launch Script - Master Node
REM Run this on the Windows machine (192.168.1.195)

set MASTER_ADDR=192.168.1.195
set MASTER_PORT=29500
set WORLD_SIZE=5
set RANK=0
set LOCAL_RANK=0

echo ================================================
echo Distributed GA+RL Training - MASTER NODE
echo ================================================
echo Master: %MASTER_ADDR%:%MASTER_PORT%
echo World size: %WORLD_SIZE% GPUs
echo ================================================

REM Check if conda environment exists
call conda activate trade 2>nul
if errorlevel 1 (
    echo Warning: 'trade' conda environment not found
    echo Trying to use current Python environment...
)

REM Check GPU
echo.
echo Checking GPU availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo Starting distributed training...
echo Waiting for workers to connect...
echo.

REM Launch training
python distributed_ga_rl.py ^
    --master_addr %MASTER_ADDR% ^
    --master_port %MASTER_PORT% ^
    --world_size %WORLD_SIZE% ^
    --rank %RANK% ^
    --local_rank %LOCAL_RANK% ^
    --data_path "F:/Market Data/Extracted" ^
    --is_master ^
    --population_size 20 ^
    --num_generations 50 ^
    --training_episodes 100

echo.
echo ================================================
echo Training complete!
echo ================================================
pause