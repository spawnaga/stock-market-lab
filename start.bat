@echo off
setlocal enabledelayedexpansion

echo Starting Stock Market Lab...

REM Load environment variables from .env if it exists
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "%%a"=="" (
                set "%%a=%%b"
            )
        )
    )
)

REM Default values if not set
if "%ENABLE_GPU%"=="" set ENABLE_GPU=true
if "%FRONTEND_PORT%"=="" set FRONTEND_PORT=3001
if "%BACKEND_API_PORT%"=="" set BACKEND_API_PORT=5100
if "%CPP_BACKEND_PORT%"=="" set CPP_BACKEND_PORT=8080
if "%GPU_SERVICE_PORT%"=="" set GPU_SERVICE_PORT=8081
if "%POSTGRES_PORT%"=="" set POSTGRES_PORT=5432
if "%REDIS_PORT%"=="" set REDIS_PORT=7379
if "%KAFKA_PORT%"=="" set KAFKA_PORT=9092
if "%MARKET_DATA_PATH%"=="" set MARKET_DATA_PATH=./data/market

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

echo ============================================
echo Configuration:
echo   Market Data Path: %MARKET_DATA_PATH%
echo   GPU Enabled: %ENABLE_GPU%
echo ============================================

REM Determine compose files based on GPU setting
set COMPOSE_FILES=-f docker-compose.yml

if /i "%ENABLE_GPU%"=="true" (
    echo Checking for NVIDIA GPU...
    nvidia-smi >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
            set GPU_NAME=%%g
        )
        if defined GPU_NAME (
            echo GPU detected: !GPU_NAME!
            set COMPOSE_FILES=!COMPOSE_FILES! -f docker-compose.gpu.yml
        ) else (
            echo WARNING: nvidia-smi available but no GPU found. Running in CPU mode.
        )
    ) else (
        echo WARNING: nvidia-smi not found. Running in CPU mode.
    )
) else (
    echo GPU disabled by configuration. Running in CPU mode.
)

REM Build and start all services
echo Building and starting all services...
docker compose %COMPOSE_FILES% up --build -d

REM Wait for services to start
echo Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Show status
echo Service Status:
docker compose ps

REM Show helpful URLs
echo.
echo Access URLs:
echo    Frontend:        http://localhost:%FRONTEND_PORT%
echo    Backend API:     http://localhost:%BACKEND_API_PORT%
echo    C++ Backend:     http://localhost:%CPP_BACKEND_PORT%
echo    GPU Service:     http://localhost:%GPU_SERVICE_PORT%
echo    Database:        localhost:%POSTGRES_PORT%
echo    Redis:           localhost:%REDIS_PORT%
echo    Kafka:           localhost:%KAFKA_PORT%
echo.
echo Use 'docker compose logs -f' to view logs
echo Use 'docker compose down' to stop all services
echo.
echo Stock Market Lab is ready!
pause
