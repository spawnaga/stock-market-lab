# Stock Market Lab - Quick Start Script for PowerShell
Write-Host "Starting Stock Market Lab..." -ForegroundColor Green

# Load environment variables from .env if it exists
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#][^=]*)=(.*)$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            if ($name -and -not [string]::IsNullOrWhiteSpace($name)) {
                [Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
    }
}

# Default values
$ENABLE_GPU = if ($env:ENABLE_GPU) { $env:ENABLE_GPU } else { "true" }
$FRONTEND_PORT = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { "3001" }
$BACKEND_API_PORT = if ($env:BACKEND_API_PORT) { $env:BACKEND_API_PORT } else { "5100" }
$CPP_BACKEND_PORT = if ($env:CPP_BACKEND_PORT) { $env:CPP_BACKEND_PORT } else { "8080" }
$GPU_SERVICE_PORT = if ($env:GPU_SERVICE_PORT) { $env:GPU_SERVICE_PORT } else { "8081" }
$POSTGRES_PORT = if ($env:POSTGRES_PORT) { $env:POSTGRES_PORT } else { "5432" }
$REDIS_PORT = if ($env:REDIS_PORT) { $env:REDIS_PORT } else { "7379" }
$KAFKA_PORT = if ($env:KAFKA_PORT) { $env:KAFKA_PORT } else { "9092" }
$MARKET_DATA_PATH = if ($env:MARKET_DATA_PATH) { $env:MARKET_DATA_PATH } else { "./data/market" }

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "Docker is running" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker is not running. Please start Docker first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Market Data Path: $MARKET_DATA_PATH" -ForegroundColor White
Write-Host "  GPU Enabled: $ENABLE_GPU" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan

# Determine compose files based on GPU setting
$composeFiles = @("-f", "docker-compose.yml")

if ($ENABLE_GPU -eq "true") {
    Write-Host "Checking for NVIDIA GPU..." -ForegroundColor Yellow
    try {
        $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($gpuInfo) {
            Write-Host "GPU detected: $gpuInfo" -ForegroundColor Green
            $composeFiles += @("-f", "docker-compose.gpu.yml")
        } else {
            Write-Host "WARNING: nvidia-smi available but no GPU found. Running in CPU mode." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "WARNING: nvidia-smi not found. Running in CPU mode." -ForegroundColor Yellow
    }
} else {
    Write-Host "GPU disabled by configuration. Running in CPU mode." -ForegroundColor Yellow
}

# Build and start all services
Write-Host "Building and starting all services..." -ForegroundColor Yellow
& docker compose $composeFiles up --build -d

# Wait for services to start
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Show status
Write-Host "Service Status:" -ForegroundColor Cyan
docker compose ps

# Show helpful URLs
Write-Host ""
Write-Host "Access URLs:" -ForegroundColor Green
Write-Host "   Frontend:        http://localhost:$FRONTEND_PORT" -ForegroundColor White
Write-Host "   Backend API:     http://localhost:$BACKEND_API_PORT" -ForegroundColor White
Write-Host "   C++ Backend:     http://localhost:$CPP_BACKEND_PORT" -ForegroundColor White
Write-Host "   GPU Service:     http://localhost:$GPU_SERVICE_PORT" -ForegroundColor White
Write-Host "   Database:        localhost:$POSTGRES_PORT" -ForegroundColor White
Write-Host "   Redis:           localhost:$REDIS_PORT" -ForegroundColor White
Write-Host "   Kafka:           localhost:$KAFKA_PORT" -ForegroundColor White
Write-Host ""
Write-Host "Use 'docker compose logs -f' to view logs" -ForegroundColor Cyan
Write-Host "Use 'docker compose down' to stop all services" -ForegroundColor Cyan
Write-Host ""
Write-Host "Stock Market Lab is ready!" -ForegroundColor Green

Read-Host "Press Enter to continue"
