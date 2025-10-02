@echo off
echo 🚀 Starting Stock Market Lab...

REM Check if Docker is running (simplified check)
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Build and start all services
echo 🔨 Building and starting all services...
docker-compose up --build -d

REM Wait for services to start
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Show status
echo 📊 Service Status:
docker-compose ps

REM Show helpful URLs
echo.
echo 🌐 Access URLs:
echo    Frontend:        http://localhost:3001
echo    Backend API:     http://localhost:8080
echo    GPU Service:     http://localhost:8081
echo    Database:        localhost:5432
echo    Redis:           localhost:6379
echo    Kafka:           localhost:9092
echo.
echo 📋 Use 'docker-compose logs -f' to view logs
echo 🛑 Use 'docker-compose down' to stop all services
echo.
echo ✅ Stock Market Lab is ready!
pause
