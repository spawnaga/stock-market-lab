# Stock Market Lab - Makefile
# Multi-language architecture management

.PHONY: help build up down logs clean dev test health

# Default target
help: ## Show this help message
	@echo "Stock Market Lab - Development Commands"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development commands
build: ## Build all services
	@echo "🔨 Building all services..."
	docker-compose build --parallel

up: ## Start all services
	@echo "🚀 Starting all services..."
	docker-compose up -d

dev: ## Start services in development mode with logs
	@echo "🔧 Starting services in development mode..."
	docker-compose up --build

down: ## Stop all services
	@echo "⏹️  Stopping all services..."
	docker-compose down

logs: ## Show logs for all services
	@echo "📋 Showing logs for all services..."
	docker-compose logs -f

clean: ## Clean up containers, images, and volumes
	@echo "🧹 Cleaning up..."
	docker-compose down -v --rmi all --remove-orphans
	docker system prune -f

# Individual service commands
frontend: ## Build and run only frontend
	@echo "🎨 Starting frontend..."
	docker-compose up --build frontend

backend: ## Build and run backend services
	@echo "⚙️  Starting backend services..."
	docker-compose up --build cpp-backend python-agents gpu-acceleration

db: ## Start only database services
	@echo "🗄️  Starting database services..."
	docker-compose up -d postgres redis kafka zookeeper

# Health and testing
health: ## Check health of all services
	@echo "🏥 Checking service health..."
	@curl -f http://localhost:8080/health || echo "❌ C++ Backend: Unhealthy"
	@curl -f http://localhost:8000/health || echo "❌ Python Agents: Unhealthy"
	@curl -f http://localhost:8081/health || echo "❌ GPU Acceleration: Unhealthy"
	@curl -f http://localhost:3001/ || echo "❌ Frontend: Unhealthy"

test: ## Run tests for all services
	@echo "🧪 Running tests..."
	@cd frontend-react && npm test -- --watchAll=false
	@cd agents-python && python -m pytest
	@cd core-cpp && make test
	@cd gpu-acceleration-rust && cargo test

# Database operations
db-init: ## Initialize database with schema
	@echo "📊 Initializing database..."
	docker-compose exec postgres psql -U stock_user -d stock_market -f /docker-entrypoint-initdb.d/schema.sql

db-reset: ## Reset database (WARNING: This will delete all data)
	@echo "⚠️  Resetting database..."
	docker-compose down postgres
	docker volume rm stock-market-lab_postgres_data || true
	docker-compose up -d postgres
	@sleep 10
	@make db-init

# Monitoring and debugging
ps: ## Show running containers
	docker-compose ps

top: ## Show running processes in containers
	docker-compose top

exec-frontend: ## Execute shell in frontend container
	docker-compose exec frontend sh

exec-backend: ## Execute shell in cpp-backend container
	docker-compose exec cpp-backend bash

exec-python: ## Execute shell in python-agents container
	docker-compose exec python-agents bash

exec-rust: ## Execute shell in gpu-acceleration container
	docker-compose exec gpu-acceleration sh

exec-db: ## Execute psql in database container
	docker-compose exec postgres psql -U stock_user -d stock_market

# Production commands
prod-build: ## Build for production
	@echo "🏭 Building for production..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

prod-up: ## Start in production mode
	@echo "🏭 Starting in production mode..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Backup and restore
backup: ## Backup database
	@echo "💾 Creating database backup..."
	docker-compose exec postgres pg_dump -U stock_user stock_market > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Development utilities
install: ## Install dependencies for all services
	@echo "📦 Installing dependencies..."
	@cd frontend-react && npm ci
	@cd agents-python && pip install -r requirements.txt
	@echo "✅ Dependencies installed"

format: ## Format code in all services
	@echo "✨ Formatting code..."
	@cd frontend-react && npm run format || true
	@cd agents-python && black . || true
	@cd core-cpp && clang-format -i src/*.cpp src/*.h || true
	@cd gpu-acceleration-rust && cargo fmt || true

lint: ## Lint code in all services
	@echo "🔍 Linting code..."
	@cd frontend-react && npm run lint || true
	@cd agents-python && flake8 . || true
	@cd gpu-acceleration-rust && cargo clippy || true

# Quick start
quick: ## Quick start (build and up)
	@echo "⚡ Quick start..."
	@make build
	@make up
	@echo "🎉 All services started!"
	@echo "🌐 Frontend: http://localhost:3001"
	@echo "🔧 Backend API: http://localhost:8080"
	@echo "🧠 GPU Service: http://localhost:8081"
