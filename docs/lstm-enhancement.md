# Enhanced LSTM Agent

## Overview

The LSTM (Long Short-Term Memory) agent has been significantly enhanced to provide more accurate price predictions through:

1. **Model Training**: The agent now trains its LSTM model on historical price data
2. **Improved Prediction Logic**: Uses both trained models and fallback methods
3. **Better Data Handling**: More robust preparation and processing of historical data
4. **Enhanced Monitoring**: Detailed status reporting including model training status

## Key Improvements

### Model Training
- LSTM model is now trained on historical price sequences
- Uses PyTorch with Adam optimizer and MSE loss function
- Trains for 50 epochs with automatic learning rate adjustment
- Model is saved and reused for subsequent predictions

### Data Processing
- Enhanced data preparation with proper sequence creation
- Support for both training and prediction data formats
- Proper scaling and normalization of price data
- Robust error handling for data inconsistencies

### Prediction Capabilities
- Dual prediction approach: trained model + fallback trend analysis
- Confidence scoring based on model performance and data quality
- Direction prediction (up/down/stable) with reasoning
- Model accuracy estimation for transparency

### Monitoring and Debugging
- New `/lstm/status` endpoint for detailed agent status
- Tracks model training status and device utilization
- Enhanced metrics collection for LSTM-specific performance

## API Endpoints

### GET `/lstm/status`
Returns detailed status information about the LSTM agent including:
- Model training status
- Device utilization (CPU/GPU)
- Sequence length configuration
- Agent metrics and performance data

## Usage

The enhanced LSTM agent automatically integrates with the existing system and provides:
- More accurate price predictions
- Better handling of insufficient data
- Improved error resilience
- Enhanced monitoring capabilities

## Configuration

The LSTM agent can be configured through:
- `sequence_length`: Number of historical data points to consider (default: 30)
- `device`: CPU or GPU for model inference (auto-detected)
- `guardrails_enabled`: Safety checks for extreme predictions

## Performance Impact

The enhanced LSTM agent introduces:
- Slight increase in initial startup time due to model training
- Improved prediction accuracy over time
- Additional memory usage for model storage
- Enhanced monitoring overhead (minimal impact)