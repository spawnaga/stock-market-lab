# GPU Accelerated Order Book Simulator

This Rust microservice provides GPU-accelerated simulations for order book computations to improve performance for complex financial calculations.

## Features

- **GPU Acceleration**: Leverages CUDA/GPU computing for high-performance simulations
- **REST API**: Provides HTTP endpoints for order book simulations
- **Real-time Processing**: Handles concurrent simulation requests efficiently
- **Health Monitoring**: Built-in health check endpoints

## API Endpoints

### POST /simulate
Submit an order book simulation request

**Request Body:**
```json
{
  "order_book": {
    "bids": [
      {
        "price": 100.5,
        "quantity": 100.0,
        "side": "Buy"
      }
    ],
    "asks": [
      {
        "price": 101.0,
        "quantity": 150.0,
        "side": "Sell"
      }
    ],
    "timestamp": 1634567890
  },
  "simulation_params": {
    "num_simulations": 1000,
    "time_horizon": 3600.0,
    "volatility": 0.02
  }
}
```

**Response:**
```json
{
  "final_price": 100.85,
  "price_trajectory": [100.5, 100.6, 100.7, ...],
  "execution_time": 0.045,
  "gpu_used": true
}
```

### GET /health
Health check endpoint

## Architecture

This service is designed to be integrated with the existing Python agents system through:
1. HTTP REST API calls
2. WebSocket communication for real-time updates
3. Shared Redis cache for intermediate results

## Building and Running

### Build
```bash
cargo build --release
```

### Run
```bash
cargo run --release
```

### Docker
```bash
docker build -t order-book-simulator .
docker run -p 8080:8080 order-book-simulator
```

## Integration with Existing System

This service integrates with the Python agents system by:
1. Receiving order book data from the agents
2. Performing GPU-accelerated simulations
3. Returning results to the agents for decision-making
4. Communicating via HTTP REST API