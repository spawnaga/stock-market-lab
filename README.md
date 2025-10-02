# AI-Driven Multi-Agent Stock Market Lab

A cutting-edge platform that combines multiple AI agents to analyze stock markets, execute trades, and provide intelligent insights through natural language queries.

## Features
- Real-time market data streaming
- Multi-agent framework with RL, LSTM, and NLP agents
- Strategy lab with natural language interface
- Visualization dashboard for trading insights
- Human-AI collaboration with guardrails
- Backtesting and forward-testing capabilities
- Advanced monitoring and performance optimization
- Real-time ROI tracking and performance analytics
- Enhanced LSTM agent with model training capabilities
- **Valuable 17+ years of 1-minute OHLCV historical data (2008-2025)** for enhanced training and backtesting

## Tech Stack
- Backend: C++ (fast data ingestion + WebSocket)
- AI/Agents: Python (PyTorch, Transformers)
- Frontend: TypeScript + React
- Data Layer: PostgreSQL + Redis
- Infrastructure: Docker + Kafka

## Performance & ROI Focus

This system is designed for **real-money trading performance** with:
- **Sub-100ms decision latency** for competitive advantage
- **Real-time ROI tracking** with comprehensive analytics
- **Performance benchmarking** to validate trading strategies
- **Risk management** with automated monitoring
- **Scalable architecture** for high-volume trading

## Running the Application

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available

### Quick Start

1. Clone the repository
```bash
git clone https://github.com/spawnaga/stock-market-lab.git
cd stock-market-lab
```

2. Start all services (choose one)
- Windows PowerShell:
```powershell
./start.ps1
```
- Windows CMD:
```bat
start.bat
```
- macOS/Linux:
```bash
chmod +x start.sh
./start.sh
```
- Or with Docker Compose directly:
```bash
docker-compose up --build
```

3. Access the dashboard
- Open your browser and go to: http://localhost:3001

### Service Ports
- Frontend Dashboard: http://localhost:3001
- C++ Backend API: http://localhost:8080
- GPU Acceleration Service: http://localhost:8081
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Kafka: localhost:9092 (internal), localhost:29092 (host)

### Development Mode
To run in development mode with live reloading (foreground logs):
```bash
docker-compose up
```

### Stopping Services
```bash
docker-compose down
```

### Database Initialization
The database will automatically initialize with the schema defined in `db/schema.sql`. 
To reset the database:
```bash
cd infra
docker-compose down
docker volume prune
docker-compose up -d
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Client   │    │   User Client   │    │   User Client   │
│   (Browser)     │    │   (Mobile)      │    │   (Terminal)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Frontend      │    │   Frontend      │
│   (React/TS)    │    │   (React/TS)    │    │   (CLI Tools)   │
│                 │    │                 │    │                 │
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

## Interactive Brokers Integration

This system supports seamless integration with Interactive Brokers (IB) for live trading capabilities:

### IB Connection Setup
1. **IB Gateway Installation**:
   - Download and install IB Gateway from Interactive Brokers website
   - Configure API access with proper credentials
   - Enable "Use SSL" and set appropriate port numbers

2. **API Configuration**:
   - Create a dedicated API user account in IB Gateway
   - Set up proper permissions for trading and market data access
   - Configure firewall rules to allow connections from the application

3. **Integration Components**:
   - **IB Adapter Module**: Python wrapper for TWS/IB Gateway API
   - **Order Management**: Real-time order placement and execution
   - **Portfolio Tracking**: Live portfolio value and position updates
   - **Risk Controls**: Position limits and risk management

### Key Features for IB Integration
- **Real-time Market Data**: Connect to IB's market data feeds
- **Paper Trading**: Test strategies without risking real capital
- **Live Trading**: Execute orders directly through IB API
- **Portfolio Management**: Track positions and PnL in real-time
- **Risk Controls**: Implement stop-losses and position sizing
- **Historical Data**: Access IB's extensive historical dataset

### Configuration Files
The system expects the following configuration for IB integration:
```yaml
ib_gateway:
  host: "127.0.0.1"
  port: 4001
  client_id: 1001
  account: "DU123456"
  use_ssl: true
```

### Security Considerations
- Store API credentials securely using environment variables
- Implement proper authentication and authorization
- Use encrypted connections for all communications
- Regular audit of trading activities and permissions