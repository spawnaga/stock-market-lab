#!/bin/bash

# Stock Market Lab - Quick Start Script
echo "🚀 Starting Stock Market Lab..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start all services
echo "🔨 Building and starting all services..."
docker-compose up --build

# Wait a bit for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Show status
echo "📊 Service Status:"
docker-compose ps

# Show helpful URLs
echo ""
echo "🌐 Access URLs:"
echo "   Frontend:        http://localhost:3001"
echo "   Backend API:     http://localhost:8080"
echo "   GPU Service:     http://localhost:8081"
echo "   Database:        localhost:5432"
echo "   Redis:           localhost:6379"
echo "   Kafka:           localhost:9092"
echo ""
echo "📋 Use 'docker-compose logs -f' to view logs"
echo "🛑 Use 'docker-compose down' to stop all services"
echo ""
echo "✅ Stock Market Lab is ready!"
