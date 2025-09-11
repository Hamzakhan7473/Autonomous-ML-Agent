'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Cog6ToothIcon, 
  ClockIcon, 
  ChartBarIcon,
  CpuChipIcon,
  InformationCircleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import { PipelineConfig } from '@/lib/types';
import { useLocalStorage } from '@/lib/hooks';

interface PipelineConfigProps {
  onConfigReady?: (config: PipelineConfig) => void;
  className?: string;
}

const PipelineConfigComponent: React.FC<PipelineConfigProps> = ({ 
  onConfigReady, 
  className = '' 
}) => {
  const [config, setConfig] = useState<PipelineConfig>({
    time_budget: 3600, // 1 hour
    optimization_metric: 'auto',
    random_state: 42,
    output_dir: './results',
    save_models: true,
    save_results: true,
    verbose: false
  });

  const [savedConfigs, setSavedConfigs] = useLocalStorage<PipelineConfig[]>('ml-pipeline-configs', []);
  const [configName, setConfigName] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string>('balanced');

  const presets = {
    quick: {
      name: 'Quick Test',
      description: 'Fast execution for testing',
      config: {
        time_budget: 300,
        optimization_metric: 'accuracy',
        random_state: 42,
        output_dir: './results',
        save_models: true,
        save_results: true,
        verbose: true
      }
    },
    balanced: {
      name: 'Balanced',
      description: 'Good balance of speed and quality',
      config: {
        time_budget: 1800,
        optimization_metric: 'auto',
        random_state: 42,
        output_dir: './results',
        save_models: true,
        save_results: true,
        verbose: false
      }
    },
    thorough: {
      name: 'Thorough',
      description: 'Maximum quality with longer runtime',
      config: {
        time_budget: 7200,
        optimization_metric: 'auto',
        random_state: 42,
        output_dir: './results',
        save_models: true,
        save_results: true,
        verbose: true
      }
    },
    custom: {
      name: 'Custom',
      description: 'User-defined configuration',
      config: config
    }
  };

  useEffect(() => {
    const preset = presets[selectedPreset as keyof typeof presets];
    if (preset && selectedPreset !== 'custom') {
      setConfig(preset.config as PipelineConfig);
    }
  }, [selectedPreset]);

  const handleConfigChange = (key: keyof PipelineConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    setSelectedPreset('custom');
  };

  const handleSaveConfig = () => {
    if (configName.trim()) {
      const newConfig = { ...config, name: configName };
      setSavedConfigs(prev => [...prev, newConfig]);
      setConfigName('');
      setShowSaveDialog(false);
    }
  };

  const handleLoadConfig = (savedConfig: PipelineConfig) => {
    setConfig(savedConfig);
    setSelectedPreset('custom');
  };

  const handleApplyConfig = () => {
    if (onConfigReady) {
      onConfigReady(config);
    }
  };

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const ConfigSection: React.FC<{
    title: string;
    icon: React.ComponentType<any>;
    children: React.ReactNode;
  }> = ({ title, icon: Icon, children }) => (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Icon className="h-5 w-5 text-gray-600" />
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      </div>
      {children}
    </div>
  );

  const ConfigField: React.FC<{
    label: string;
    description?: string;
    children: React.ReactNode;
  }> = ({ label, description, children }) => (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        {label}
      </label>
      {description && (
        <p className="text-sm text-gray-500">{description}</p>
      )}
      {children}
    </div>
  );

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900">Pipeline Configuration</h2>
        <p className="mt-2 text-gray-600">
          Configure your machine learning pipeline parameters
        </p>
      </div>

      {/* Presets */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg border border-gray-200 p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Presets</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(presets).map(([key, preset]) => (
            <button
              key={key}
              onClick={() => setSelectedPreset(key)}
              className={`p-4 rounded-lg border-2 text-left transition-colors ${
                selectedPreset === key
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <h4 className="font-medium text-gray-900">{preset.name}</h4>
              <p className="text-sm text-gray-600 mt-1">{preset.description}</p>
              {key !== 'custom' && (
                <p className="text-xs text-gray-500 mt-2">
                  Time: {formatTime((preset.config as PipelineConfig).time_budget)}
                </p>
              )}
            </button>
          ))}
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Execution Settings */}
        <ConfigSection title="Execution Settings" icon={ClockIcon}>
          <div className="space-y-6">
            <ConfigField
              label="Time Budget"
              description="Maximum time allowed for training (seconds)"
            >
              <div className="space-y-2">
                <input
                  type="range"
                  min="300"
                  max="14400"
                  step="300"
                  value={config.time_budget}
                  onChange={(e) => handleConfigChange('time_budget', parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-sm text-gray-600">
                  <span>5m</span>
                  <span className="font-medium">{formatTime(config.time_budget)}</span>
                  <span>4h</span>
                </div>
              </div>
            </ConfigField>

            <ConfigField
              label="Optimization Metric"
              description="Primary metric for model selection"
            >
              <select
                value={config.optimization_metric}
                onChange={(e) => handleConfigChange('optimization_metric', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="auto">Auto (detect from data)</option>
                <option value="accuracy">Accuracy</option>
                <option value="precision">Precision</option>
                <option value="recall">Recall</option>
                <option value="f1_score">F1 Score</option>
                <option value="roc_auc">ROC AUC</option>
                <option value="mse">Mean Squared Error</option>
                <option value="mae">Mean Absolute Error</option>
                <option value="r2">R² Score</option>
              </select>
            </ConfigField>

            <ConfigField
              label="Random State"
              description="Seed for reproducible results"
            >
              <input
                type="number"
                value={config.random_state}
                onChange={(e) => handleConfigChange('random_state', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                min="0"
                max="999999"
              />
            </ConfigField>
          </div>
        </ConfigSection>

        {/* Output Settings */}
        <ConfigSection title="Output Settings" icon={ChartBarIcon}>
          <div className="space-y-6">
            <ConfigField
              label="Output Directory"
              description="Directory to save results and models"
            >
              <input
                type="text"
                value={config.output_dir}
                onChange={(e) => handleConfigChange('output_dir', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="./results"
              />
            </ConfigField>

            <div className="space-y-4">
              <ConfigField
                label="Save Options"
                description="Choose what to save after training"
              >
                <div className="space-y-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={config.save_models}
                      onChange={(e) => handleConfigChange('save_models', e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-sm text-gray-700">Save trained models</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={config.save_results}
                      onChange={(e) => handleConfigChange('save_results', e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-sm text-gray-700">Save detailed results</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={config.verbose}
                      onChange={(e) => handleConfigChange('verbose', e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <span className="ml-2 text-sm text-gray-700">Verbose logging</span>
                  </label>
                </div>
              </ConfigField>
            </div>
          </div>
        </ConfigSection>
      </div>

      {/* Saved Configurations */}
      {savedConfigs.length > 0 && (
        <ConfigSection title="Saved Configurations" icon={CpuChipIcon}>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {savedConfigs.map((savedConfig, index) => (
              <div key={index} className="p-4 border border-gray-200 rounded-lg">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium text-gray-900">
                      {(savedConfig as any).name || `Config ${index + 1}`}
                    </h4>
                    <p className="text-sm text-gray-600">
                      Time: {formatTime(savedConfig.time_budget)}
                    </p>
                    <p className="text-sm text-gray-600">
                      Metric: {savedConfig.optimization_metric}
                    </p>
                  </div>
                  <button
                    onClick={() => handleLoadConfig(savedConfig)}
                    className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                  >
                    Load
                  </button>
                </div>
              </div>
            ))}
          </div>
        </ConfigSection>
      )}

      {/* Save Configuration Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Save Configuration</h3>
            <input
              type="text"
              value={configName}
              onChange={(e) => setConfigName(e.target.value)}
              placeholder="Enter configuration name"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 mb-4"
            />
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowSaveDialog(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveConfig}
                disabled={!configName.trim()}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-between items-center bg-white rounded-lg border border-gray-200 p-6"
      >
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setShowSaveDialog(true)}
            className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
          >
            <Cog6ToothIcon className="h-4 w-4" />
            <span>Save Configuration</span>
          </button>
        </div>
        <button
          onClick={handleApplyConfig}
          className="flex items-center space-x-2 px-6 py-3 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
        >
          <CheckCircleIcon className="h-4 w-4" />
          <span>Apply Configuration</span>
        </button>
      </motion.div>

      {/* Info Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-blue-50 border border-blue-200 rounded-lg p-4"
      >
        <div className="flex items-start space-x-2">
          <InformationCircleIcon className="h-5 w-5 text-blue-500 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-blue-800">Configuration Tips</h3>
            <ul className="mt-1 text-sm text-blue-700 space-y-1">
              <li>• Longer time budgets allow for more thorough hyperparameter optimization</li>
              <li>• "Auto" metric selection works best for most datasets</li>
              <li>• Save models to enable predictions after training</li>
              <li>• Verbose logging provides detailed training information</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default PipelineConfigComponent;