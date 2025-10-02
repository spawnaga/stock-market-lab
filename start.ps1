# Stock Market Lab - Quick Start Script for PowerShell
Write-Host "🚀 Starting Stock Market Lab..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Build and start all services
Write-Host "🔨 Building and starting all services..." -ForegroundColor Yellow
docker-compose up --build -d

# Wait for services to start
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Show status
Write-Host "📊 Service Status:" -ForegroundColor Cyan
docker-compose ps

# Show helpful URLs
Write-Host ""
Write-Host "🌐 Access URLs:" -ForegroundColor Green
Write-Host "   Frontend:        http://localhost:3001" -ForegroundColor White
Write-Host "   Backend API:     http://localhost:8080" -ForegroundColor White
Write-Host "   GPU Service:     http://localhost:8081" -ForegroundColor White
Write-Host "   Database:        localhost:5432" -ForegroundColor White
Write-Host "   Redis:           localhost:6379" -ForegroundColor White
Write-Host "   Kafka:           localhost:9092" -ForegroundColor White
Write-Host ""
Write-Host "📋 Use 'docker-compose logs -f' to view logs" -ForegroundColor Cyan
Write-Host "🛑 Use 'docker-compose down' to stop all services" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Stock Market Lab is ready!" -ForegroundColor Green

Read-Host "Press Enter to continue"
