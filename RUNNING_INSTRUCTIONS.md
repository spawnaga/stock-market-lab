# How to Run the Stock Market Lab Application

## Overview
This is a complex multi-service application combining:
- React/TypeScript frontend
- C++ backend for data ingestion  
- Python agents for AI/ML processing
- PostgreSQL database
- Redis cache
- Kafka message broker

## Prerequisites

### System Requirements
- Docker and Docker Compose installed
- At least 4GB RAM available
- Node.js >= 16.x (for frontend development)
- Python 3.9+ (for agents)

### Current Environment Status
```
$ docker --version
Docker version 28.3.3, build 980b856

$ docker compose version
Docker Compose version v2.39.1

$ docker info
Docker is not running properly
```

## Installation Steps

### 1. Install Docker and Docker Compose
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose

# Or use official installation
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER
```

### 2. Start Docker Service
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### 3. Clone and Prepare Repository
```bash
git clone https://github.com/spawnaga/stock-market-lab.git
cd stock-market-lab
```

### 4. Install Python Dependencies (for agents)
```bash
cd agents-python
pip install -r requirements.txt
```

### 5. Run the Application
```bash
cd infra
docker-compose up -d
```

## Expected Services
Once running, these services should be available:
- Frontend Dashboard: http://localhost:3001
- C++ Backend: http://localhost:8080  
- Python Agents: http://localhost:5000
- PostgreSQL: localhost:5432
- Redis: localhost:6379

## Troubleshooting

### If Docker fails to start:
1. Check if Docker daemon is running:
   ```bash
   sudo systemctl status docker
   ```

2. Restart Docker:
   ```bash
   sudo systemctl restart docker
   ```

3. Check Docker permissions:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

### If frontend fails to build:
1. Install Node.js dependencies:
   ```bash
   cd frontend-react
   npm install
   ```

2. Check for missing dependencies:
   ```bash
   npm install react-router-dom web-vitals
   ```

## Fixed Issues in Source Code

### 1. Syntax Error in StrategyLab.tsx
**Problem:** Missing closing parenthesis
```diff
- setIsGenerating(true;
+ setIsGenerating(true);
```

### 2. TypeScript Configuration Issues
**Problem:** Invalid tsconfig.json flags
**Solution:** Changed `"moduleResolution": "bundler"` to `"moduleResolution": "node"`

## Known Issues That Need Attention

1. **Missing react-router-dom dependency** - Needed for navigation
2. **Missing web-vitals dependency** - Used in reportWebVitals.ts
3. **Incomplete frontend components** - Some components may be missing or incomplete
4. **Database schema** - Needs proper initialization scripts
5. **Backend service connectivity** - May need network configuration adjustments

## Testing the Application

After successful deployment:
1. Visit http://localhost:3001 for the dashboard
2. Check that all microservices are healthy
3. Verify database connection
4. Test agent communication channels