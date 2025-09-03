import React, { useState, useEffect } from 'react';
import './StrategyLab.css';

interface StrategyForm {
  name: string;
  description: string;
  parameters: {
    [key: string]: any;
  };
}

const StrategyLab: React.FC = () => {
  const [strategyName, setStrategyName] = useState('');
  const [strategyDescription, setStrategyDescription] = useState('');
  const [strategyParameters, setStrategyParameters] = useState<{[key: string]: any}>({});
  const [generatedStrategies, setGeneratedStrategies] = useState<any[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<any>(null);

  // Mock data for existing strategies
  useEffect(() => {
    setGeneratedStrategies([
      {
        id: 'strat-001',
        name: 'Momentum Trader',
        description: 'Buys stocks showing strong upward momentum',
        parameters: {
          lookback_period: 14,
          threshold: 0.02,
          max_positions: 5
        },
        createdAt: '2025-08-24'
      },
      {
        id: 'strat-002',
        name: 'Mean Reversion',
        description: 'Trades against extreme price deviations',
        parameters: {
          lookback_period: 30,
          deviation_threshold: 2.0,
          position_size: 0.1
        },
        createdAt: '2025-08-23'
      }
    ]);
  }, []);

  const handleParameterChange = (paramName: string, value: any) => {
    setStrategyParameters(prev => ({
      ...prev,
      [paramName]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);

    // Simulate API call to backend
    setTimeout(() => {
      const newStrategy = {
        id: `strat-${Date.now()}`,
        name: strategyName,
        description: strategyDescription,
        parameters: strategyParameters,
        createdAt: new Date().toISOString().split('T')[0]
      };

      setGeneratedStrategies(prev => [newStrategy, ...prev]);
      setStrategyName('');
      setStrategyDescription('');
      setStrategyParameters({});
      setIsGenerating(false);
    }, 1500);
  };

  const handleStrategySelect = (strategy: any) => {
    setSelectedStrategy(strategy);
  };

  return (
    <div className="strategy-lab">
      <div className="strategy-lab-header">
        <h2>Strategy Lab</h2>
        <p>Create and manage trading strategies using natural language</p>
      </div>

      <div className="strategy-lab-content">
        <div className="strategy-form-section">
          <h3>Create New Strategy</h3>
          <form onSubmit={handleSubmit} className="strategy-form">
            <div className="form-group">
              <label htmlFor="strategyName">Strategy Name:</label>
              <input
                type="text"
                id="strategyName"
                value={strategyName}
                onChange={(e) => setStrategyName(e.target.value)}
                placeholder="e.g., Momentum Trader"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="strategyDescription">Description:</label>
              <textarea
                id="strategyDescription"
                value={strategyDescription}
                onChange={(e) => setStrategyDescription(e.target.value)}
                placeholder="Describe your strategy in natural language..."
                rows={3}
                required
              />
            </div>

            <div className="form-group">
              <label>Parameters:</label>
              <div className="parameters-grid">
                <div className="parameter-item">
                  <label htmlFor="lookbackPeriod">Lookback Period:</label>
                  <input
                    type="number"
                    id="lookbackPeriod"
                    value={strategyParameters.lookback_period || ''}
                    onChange={(e) => handleParameterChange('lookback_period', parseInt(e.target.value))}
                    min="1"
                    max="365"
                  />
                </div>
                <div className="parameter-item">
                  <label htmlFor="threshold">Threshold:</label>
                  <input
                    type="number"
                    id="threshold"
                    value={strategyParameters.threshold || ''}
                    onChange={(e) => handleParameterChange('threshold', parseFloat(e.target.value))}
                    step="0.01"
                    min="0"
                  />
                </div>
                <div className="parameter-item">
                  <label htmlFor="maxPositions">Max Positions:</label>
                  <input
                    type="number"
                    id="maxPositions"
                    value={strategyParameters.max_positions || ''}
                    onChange={(e) => handleParameterChange('max_positions', parseInt(e.target.value))}
                    min="1"
                    max="20"
                  />
                </div>
              </div>
            </div>

            <button type="submit" disabled={isGenerating} className="submit-btn">
              {isGenerating ? 'Generating...' : 'Create Strategy'}
            </button>
          </form>
        </div>

        <div className="strategy-list-section">
          <h3>Generated Strategies</h3>
          <div className="strategies-grid">
            {generatedStrategies.map((strategy) => (
              <div 
                key={strategy.id} 
                className={`strategy-card ${selectedStrategy?.id === strategy.id ? 'selected' : ''}`}
                onClick={() => handleStrategySelect(strategy)}
              >
                <h4>{strategy.name}</h4>
                <p className="strategy-description">{strategy.description}</p>
                <div className="strategy-meta">
                  <span className="created-at">Created: {strategy.createdAt}</span>
                </div>
                <div className="strategy-params">
                  {Object.entries(strategy.parameters).map(([key, value]) => (
                    <span key={key} className="param-tag">
                      {key}: {value}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {selectedStrategy && (
        <div className="strategy-details-section">
          <h3>Strategy Details</h3>
          <div className="strategy-details">
            <div className="detail-row">
              <label>Name:</label>
              <span>{selectedStrategy.name}</span>
            </div>
            <div className="detail-row">
              <label>Description:</label>
              <span>{selectedStrategy.description}</span>
            </div>
            <div className="detail-row">
              <label>Created:</label>
              <span>{selectedStrategy.createdAt}</span>
            </div>
            <div className="detail-row">
              <label>Parameters:</label>
              <div className="parameters-list">
                {Object.entries(selectedStrategy.parameters).map(([key, value]) => (
                  <div key={key} className="param-item">
                    <span className="param-name">{key}:</span>
                    <span className="param-value">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StrategyLab;