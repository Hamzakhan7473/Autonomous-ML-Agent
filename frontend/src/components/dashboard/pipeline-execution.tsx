'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  PlayIcon, 
  PauseIcon, 
  StopIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { usePipelineExecution, useTaskMonitor } from '@/lib/hooks';
import { PipelineConfig, PipelineTask } from '@/lib/types';
import { formatDuration } from '@/lib/api';

interface PipelineExecutionProps {
  onExecutionComplete?: (taskId: string) => void;
  className?: string;
}

const PipelineExecutionComponent: React.FC<PipelineExecutionProps> = ({ 
  onExecutionComplete, 
  className = '' 
}) => {
  const [config, setConfig] = useState<PipelineConfig | null>(null);
  const [datasetPath, setDatasetPath] = useState<string>('');
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [isStarted, setIsStarted] = useState(false);
  
  const { executing, taskId, error, executePipeline, reset } = usePipelineExecution();
  const { status, loading: monitoringLoading } = useTaskMonitor(taskId);

  // Load configuration from localStorage or use default
  useEffect(() => {
    const savedConfig = localStorage.getItem('ml-pipeline-config');
    const savedDataset = localStorage.getItem('ml-dataset-path');
    const savedTarget = localStorage.getItem('ml-target-column');
    
    if (savedConfig) {
      setConfig(JSON.parse(savedConfig));
    }
    if (savedDataset) {
      setDatasetPath(savedDataset);
    }
    if (savedTarget) {
      setTargetColumn(savedTarget);
    }
  }, []);

  useEffect(() => {
    if (taskId && !executing) {
      setIsStarted(true);
    }
  }, [taskId, executing]);

  useEffect(() => {
    if (status?.status === 'completed' && onExecutionComplete) {
      onExecutionComplete(taskId || '');
    }
  }, [status?.status, taskId, onExecutionComplete]);

  const handleStartExecution = async () => {
    if (!config || !datasetPath || !targetColumn) {
      alert('Please configure the pipeline and upload a dataset first.');
      return;
    }

    try {
      await executePipeline(config, datasetPath, targetColumn);
    } catch (err) {
      console.error('Pipeline execution failed:', err);
    }
  };

  const handleStopExecution = () => {
    // Note: This would require a stop endpoint in the backend
    console.log('Stopping pipeline execution...');
    reset();
    setIsStarted(false);
  };

  const handleReset = () => {
    reset();
    setIsStarted(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-100';
      case 'running':
        return 'text-blue-600 bg-blue-100';
      case 'failed':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return CheckCircleIcon;
      case 'running':
        return ArrowPathIcon;
      case 'failed':
        return ExclamationTriangleIcon;
      default:
        return ClockIcon;
    }
  };

  const renderExecutionStatus = () => {
    if (!isStarted && !taskId) return null;

    const StatusIcon = status ? getStatusIcon(status.status) : ClockIcon;
    const statusColor = status ? getStatusColor(status.status) : 'text-gray-600 bg-gray-100';

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg border border-gray-200 p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-full ${statusColor}`}>
              <StatusIcon className="h-6 w-6" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                {status?.status === 'running' ? 'Training in Progress' : 
                 status?.status === 'completed' ? 'Training Complete' :
                 status?.status === 'failed' ? 'Training Failed' : 'Initializing...'}
              </h3>
              <p className="text-sm text-gray-600">
                Task ID: {taskId}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600">Progress</p>
            <p className="text-2xl font-bold text-gray-900">
              {status ? Math.round(status.progress * 100) : 0}%
            </p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-4">
          <div className="w-full bg-gray-200 rounded-full h-3">
            <motion.div
              className="bg-blue-500 h-3 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${status ? status.progress * 100 : 0}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>

        {/* Status Message */}
        {status?.message && (
          <div className="mb-4">
            <p className="text-sm text-gray-700">{status.message}</p>
          </div>
        )}

        {/* Error Message */}
        {status?.error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-center space-x-2">
              <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
              <p className="text-sm text-red-700">{status.error}</p>
            </div>
          </div>
        )}

        {/* Estimated Time */}
        {status?.status === 'running' && config && (
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <div className="flex items-center space-x-1">
              <ClockIcon className="h-4 w-4" />
              <span>Estimated remaining: {formatDuration(config.time_budget * (1 - status.progress))}</span>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3 mt-4">
          {status?.status === 'running' && (
            <button
              onClick={handleStopExecution}
              className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-red-700 bg-red-100 rounded-md hover:bg-red-200"
            >
              <StopIcon className="h-4 w-4" />
              <span>Stop</span>
            </button>
          )}
          
          {(status?.status === 'completed' || status?.status === 'failed') && (
            <button
              onClick={handleReset}
              className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
            >
              <ArrowPathIcon className="h-4 w-4" />
              <span>Reset</span>
            </button>
          )}
        </div>
      </motion.div>
    );
  };

  const renderConfiguration = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg border border-gray-200 p-6"
    >
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Execution Configuration</h3>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Dataset Path
          </label>
          <input
            type="text"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            placeholder="Path to your dataset file"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Target Column
          </label>
          <input
            type="text"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            placeholder="Name of the target column"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {config && (
          <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-200">
            <div>
              <p className="text-sm font-medium text-gray-700">Time Budget</p>
              <p className="text-sm text-gray-600">{formatDuration(config.time_budget)}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700">Optimization Metric</p>
              <p className="text-sm text-gray-600">{config.optimization_metric}</p>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );

  const renderError = () => {
    if (!error) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-red-50 border border-red-200 rounded-lg p-4"
      >
        <div className="flex items-center space-x-2">
          <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
          <h3 className="text-sm font-medium text-red-800">Execution Failed</h3>
        </div>
        <p className="mt-1 text-sm text-red-700">{error}</p>
        <button
          onClick={handleReset}
          className="mt-2 text-sm text-red-600 hover:text-red-800 underline"
        >
          Try again
        </button>
      </motion.div>
    );
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900">Execute Pipeline</h2>
        <p className="mt-2 text-gray-600">
          Start the autonomous machine learning pipeline
        </p>
      </div>

      {/* Configuration */}
      {!isStarted && renderConfiguration()}

      {/* Error */}
      {error && renderError()}

      {/* Execution Status */}
      {renderExecutionStatus()}

      {/* Start Button */}
      {!isStarted && !executing && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <button
            onClick={handleStartExecution}
            disabled={!config || !datasetPath || !targetColumn}
            className="inline-flex items-center space-x-2 px-8 py-4 text-lg font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <PlayIcon className="h-6 w-6" />
            <span>Start Pipeline Execution</span>
          </button>
          
          {(!config || !datasetPath || !targetColumn) && (
            <p className="mt-2 text-sm text-gray-500">
              Please configure the pipeline and upload a dataset first
            </p>
          )}
        </motion.div>
      )}

      {/* Loading State */}
      {executing && !taskId && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-8"
        >
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-sm text-gray-600">Starting pipeline execution...</p>
        </motion.div>
      )}

      {/* Info Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-blue-50 border border-blue-200 rounded-lg p-4"
      >
        <div className="flex items-start space-x-2">
          <InformationCircleIcon className="h-5 w-5 text-blue-500 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-blue-800">Execution Information</h3>
            <ul className="mt-1 text-sm text-blue-700 space-y-1">
              <li>• The pipeline will automatically select and optimize multiple ML models</li>
              <li>• Progress updates will be shown in real-time</li>
              <li>• You can stop the execution at any time</li>
              <li>• Results will be available once training is complete</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default PipelineExecutionComponent;