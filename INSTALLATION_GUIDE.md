# Complete Installation Guide for AI-Driven Multi-Agent Stock Market Lab

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Configuration](#configuration)
5. [Data Integration](#data-integration)
6. [Running the Application](#running-the-application)
7. [Using the System](#using-the-system)
8. [Troubleshooting](#troubleshooting)
9. [System Architecture](#system-architecture)
10. [Key Features](#key-features)

## Overview

The AI-Driven Multi-Agent Stock Market Lab is a sophisticated trading platform that combines multiple AI agents (Reinforcement Learning, LSTM, News/NLP) to analyze market data, make trading decisions, and provide intelligent insights through a web-based dashboard.

## Prerequisites

### Hardware Requirements
- **CPU**: Minimum 4-core processor
- **RAM**: Minimum 8GB RAM (16GB recommended)
- **Storage**: Minimum 50GB free space
- **Network**: Stable internet connection

### Software Requirements
- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- **Git** (version 2.0 or higher)
- **Python 3.8+** (for development/testing)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/spawnaga/stock-market-lab.git
cd stock-market-lab
```

### 2. Verify Directory Structure

The repository should have the following structure:
```
stock-market-lab/
├── infra/                    # Docker configuration
│   └── docker-compose.yml
├── agents-python/           # Python agents (LSTM, RL, News/NLP)
├── core-cpp/               # C++ backend for data ingestion
├── frontend-react/         # React/TypeScript frontend
├── gpu-acceleration-rust/  # Rust GPU acceleration service
├── db/                     # Database schemas
├── docs/                   # Documentation
├── DATA_LOADING_README.md  # Data loading guide
├── data_loading_script.py  # Data loading utility
├── validate_data.py        # Data validation utility
└── README.md              # Main documentation
```

### 3. Install Docker and Docker Compose

#### Ubuntu/Debian:
```bash
# Update package index
sudo apt update

# Install Docker
sudo apt install docker.io

# Install Docker Compose
sudo apt install docker-compose

# Add current user to docker group
sudo usermod -aG docker $USER

# Re-login or run: newgrp docker
```

#### CentOS/RHEL/Fedora:
```bash
# Install Docker
sudo yum install docker

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### macOS:
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
```

#### Windows:
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop
```

### 4. Verify Installation

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Test Docker installation
docker run hello-world
```

## Configuration

### 1. Environment Variables

Create `.env` file in the root directory (optional, defaults are provided):

```bash
# Database configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=stock_market
DB_USER=stock_user
DB_PASSWORD=stock_password

# Redis configuration
REDIS_HOST=redis
REDIS_PORT=6379

# API configuration
SECRET_KEY=your-super-secret-key-here
JWT_EXPIRATION_SECONDS=3600
```

### 2. Docker Compose Configuration

The main configuration is in `infra/docker-compose.yml`. This file orchestrates all services:

- **PostgreSQL**: Database for storing market data and application state
- **Redis**: Cache and message broker for real-time communication
- **Kafka**: Message queue for data streaming
- **C++ Backend**: High-performance data ingestion and processing
- **Python Agents**: AI agents (LSTM, RL, News/NLP)
- **GPU Acceleration**: Rust service for order book simulations
- **Frontend**: React dashboard for visualization

## Data Integration

### 1. Data Format Requirements

Your historical data should be in CSV format with these columns:
```
timestamp,open,high,low,close,volume
2008-01-02 06:00:00,3352.5,3353.75,3351.75,3353.25,184
2008-01-02 06:01:00,3353.25,3354.25,3353.25,3354.25,83
```

### 2. Loading Your Data

#### Using the Data Loading Script

```bash
# Load your data (replace with your actual file path)
python data_loading_script.py --data-file /path/to/your/historical_data.csv --symbol NQ

# Example with sample data
python data_loading_script.py --data-file nq_sample_data.csv --symbol NQ
```

#### Validate Data Loading

```bash
# Verify data was loaded correctly
python validate_data.py
```

### 3. Data Storage Structure

The system stores data in Redis with these keys:
- `historical_prices`: General list of all historical data points
- `historical_prices_{SYMBOL}`: Symbol-specific historical data (e.g., `historical_prices_NQ`)

## Running the Application

### 1. Start All Services

```bash
# Navigate to the infra directory
cd infra

# Start all services in detached mode
docker-compose up -d

# Or start with verbose output for debugging
docker-compose up
```

### 2. Check Service Status

```bash
# View running containers
docker-compose ps

# View logs for a specific service
docker-compose logs python-agents

# Follow logs in real-time
docker-compose logs -f python-agents
```

### 3. Stop All Services

```bash
# Navigate to infra directory
cd infra

# Stop all services
docker-compose down

# Stop and remove volumes (reset database)
docker-compose down -v
```

## Using the System

### 1. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:3001
```

### 2. Key Features

#### A. Multi-Agent Trading System
- **Reinforcement Learning Agent**: Makes trading decisions based on market conditions
- **LSTM Price Predictor**: Predicts future price movements using neural networks
- **News/NLP Agent**: Analyzes news sentiment for market insights

#### B. Strategy Lab
- Natural language interface for creating trading strategies
- Parameter configuration for different strategy types
- Real-time strategy management and display

#### C. Real-Time Dashboard
- Live market data visualization
- Agent decision tracking
- Performance metrics and analytics

#### D. Backtesting Framework
- Comprehensive backtesting capabilities
- Test strategies against historical data
- Performance comparison and optimization

#### E. ROI Analytics
- Advanced performance metrics and return tracking
- Risk analysis and portfolio management
- Detailed reporting and visualization

### 3. API Endpoints

#### Agent Management
- `GET /lstm/status` - Get LSTM agent status
- `POST /lstm/retrain` - Manually retrain LSTM model
- `PUT /guardrails/{agent_id}/{setting}` - Toggle guardrails

#### Data Access
- `GET /metrics` - System performance metrics
- `GET /health` - Health check endpoint
- `GET /strategies` - Available trading strategies

#### Backtesting
- `POST /backtest` - Run a backtest
- `POST /compare-strategies` - Compare multiple strategies

## Troubleshooting

### Common Issues and Solutions

#### 1. Port Conflicts
If port 3001 is already in use:
```bash
# Change the port in infra/docker-compose.yml
# Then rebuild the frontend service
cd infra
docker-compose build frontend
docker-compose up -d
```

#### 2. Docker Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Re-login or run:
newgrp docker
```

#### 3. Database Initialization Problems
```bash
# Reset database and start fresh
cd infra
docker-compose down -v
docker-compose up -d
```

#### 4. Slow Startup
```bash
# Check service logs for issues
docker-compose logs

# Increase system resources if needed
```

#### 5. Data Loading Issues
```bash
# Validate your CSV format
# Check that timestamps are in correct format
# Ensure all required fields are present
```

### Debugging Commands

```bash
# Check system resources
free -h
df -h

# Check Docker resources
docker stats

# View container logs
docker-compose logs -f

# Inspect specific container
docker inspect <container_name>
```

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Client   │    │   User Client   │    │   User Client   │
│   (Browser)     │    │   (Mobile)      │    │   (Terminal)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Frontend      │    │   Frontend      │
│   (React/TS)    │    │   (React/TS)    │    │   Frontend      │
│                 │    │                 │    │   (CLI Tools)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │   API Gateway   │    │   API Gateway   │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   C++ Backend   │    │   C++ Backend   │    │   C++ Backend   │
│   (Data Ingest) │    │   (Data Ingest) │    │   (Data Ingest) │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Kafka Bus     │    │   Kafka Bus     │    │   Kafka Bus     │
│   (Messaging)   │    │   (Messaging)   │    │   (Messaging)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Python Agents │    │   Python Agents │    │   Python Agents │
│   (RL/LSTM/NLP) │    │   (RL/LSTM/NLP) │    │   (RL/LSTM/NLP) │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Redis Cache   │    │   PostgreSQL    │
│   (Storage)     │    │   (Cache)       │    │   (Storage)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IB Gateway    │    │   IB Gateway    │    │   IB Gateway    │
│   (Interactive) │    │   (Interactive) │    │   (Interactive) │
│   Connectors    │    │   Connectors    │    │   Connectors    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

### 1. Multi-Agent Architecture
- **Reinforcement Learning Agent**: Learns optimal trading strategies through experience
- **LSTM Price Predictor**: Neural network for price forecasting
- **News/NLP Agent**: Analyzes sentiment from financial news sources

### 2. Advanced Analytics
- **Real-time ROI Tracking**: Monitor performance metrics
- **Backtesting Framework**: Validate strategies against historical data
- **Risk Management**: Automated risk controls and position sizing

### 3. Human-AI Collaboration
- **Guardrails and Override Mechanisms**: Human oversight for critical decisions
- **Natural Language Interface**: Create strategies using plain English
- **Interactive Dashboard**: Real-time monitoring and control

### 4. Scalability Features
- **Microservices Architecture**: Independent scalable components
- **Distributed Processing**: Parallel execution of agents
- **Resource Monitoring**: Performance optimization and alerting

### 5. Security & Reliability
- **JWT Authentication**: Secure API access
- **Rate Limiting**: Protection against abuse
- **Graceful Shutdown**: Safe service termination
- **Logging & Monitoring**: Comprehensive system diagnostics

## Next Steps

1. **Load Your Data**: Use the data loading utilities with your 17+ years of historical data
2. **Start the System**: Run `cd infra && docker-compose up -d`
3. **Access Dashboard**: Visit http://localhost:3001
4. **Explore Features**: Test strategies, monitor agents, and analyze performance

Your 17+ years of NQ futures data will now power the LSTM agent and enable comprehensive backtesting across multiple market conditions, making your trading strategies much more robust and validated against historical performance.