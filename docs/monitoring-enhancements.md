# Enhanced Monitoring and Performance Optimization

This document describes the enhanced monitoring and performance optimization features added to the AI-Driven Multi-Agent Stock Market Lab.

## New Monitoring Capabilities

### 1. Detailed Agent Metrics
- **Execution Times**: Tracks detailed execution times for each agent
- **Success Rates**: Calculates and stores success rates for each agent
- **Error Tracking**: Records detailed error information with timestamps
- **Performance History**: Maintains recent execution times for trend analysis

### 2. System-Level Metrics
- **Memory Usage**: Monitors memory consumption in MB
- **CPU Utilization**: Tracks CPU percentage usage
- **Thread Count**: Monitors active thread count
- **Connected Clients**: Tracks WebSocket client connections

### 3. Request-Level Metrics
- **Request Timing**: Tracks request processing times
- **Requests Per Second**: Calculates throughput metrics
- **Error Statistics**: Counts and tracks request errors

## New API Endpoints

### `/metrics` (GET)
Returns comprehensive system metrics including:
- Agent-specific performance data
- System resource usage
- Request statistics
- Process information

### `/health` (GET)
Enhanced health check with:
- Detailed system resource information
- Performance metrics
- Agent success rates
- Request throughput data

### `/performance/optimization` (GET)
Provides performance optimization recommendations based on:
- Memory usage thresholds
- CPU utilization levels
- Thread count monitoring
- Request latency analysis
- Agent success rate evaluation

## Performance Improvements

### 1. Enhanced Error Handling
- Improved error tracking with detailed error messages
- Error details stored for debugging purposes
- Better exception propagation and logging

### 2. Resource Management
- Optimized memory usage patterns
- Thread count monitoring for concurrency control
- Efficient data structure usage

### 3. Scalability Enhancements
- Better handling of concurrent requests
- Improved resource allocation strategies
- Performance monitoring for scaling decisions

## Usage Examples

### Getting System Metrics
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" http://localhost:5000/metrics
```

### Health Check
```bash
curl http://localhost:5000/health
```

### Performance Recommendations
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" http://localhost:5000/performance/optimization
```

## Monitoring Best Practices

1. **Regular Monitoring**: Set up automated monitoring of `/health` endpoint
2. **Alerting**: Configure alerts based on thresholds in `/performance/optimization`
3. **Performance Tuning**: Use `/metrics` and `/performance/optimization` for tuning
4. **Error Analysis**: Review error details in agent metrics for troubleshooting

## Thresholds and Alerts

| Metric | Warning Threshold | Critical Threshold |
|--------|------------------|-------------------|
| Memory Usage | 500 MB | 1000 MB |
| CPU Usage | 80% | 90% |
| Active Threads | 50 | 100 |
| Request Latency | 1.0 sec | 5.0 sec |
| Agent Success Rate | < 80% | < 50% |