'use client';

import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { 
  CloudArrowUpIcon, 
  DocumentIcon,
  SparklesIcon,
  InformationCircleIcon,
  PlayIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { PipelineConfig } from '@/lib/types';

interface PipelineConfigProps {
  onStartPipeline?: (file: File, config: PipelineConfig) => void;
  className?: string;
}

const PipelineConfigComponent: React.FC<PipelineConfigProps> = ({ 
  onStartPipeline, 
  className = '' 
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [config, setConfig] = useState<PipelineConfig>({
    time_budget: 1800, // 30 minutes - good balance
    optimization_metric: 'auto',
    random_state: 42,
    output_dir: './results',
    save_models: true,
    save_results: true,
    verbose: false
  });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (file && file.type === 'text/csv') {
      setSelectedFile(file);
    } else {
      alert('Please select a CSV file');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleStartPipeline = () => {
    if (selectedFile && onStartPipeline) {
      onStartPipeline(selectedFile, config);
    }
  };

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900">Start ML Pipeline</h2>
        <p className="mt-2 text-gray-600">
          Upload your dataset and let AI automatically configure the pipeline
        </p>
      </div>

      {/* File Upload Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg border border-gray-200 p-8"
      >
        <div className="text-center">
          <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-semibold text-gray-900">Upload Dataset</h3>
          <p className="mt-2 text-sm text-gray-600">
            Drag and drop your CSV file here, or click to browse
          </p>
        </div>

        <div
          className={`mt-6 border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            isDragOver
              ? 'border-blue-500 bg-blue-50'
              : selectedFile
              ? 'border-green-500 bg-green-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {selectedFile ? (
            <div className="flex items-center justify-center space-x-3">
              <DocumentIcon className="h-8 w-8 text-green-600" />
              <div>
                <p className="text-sm font-medium text-green-900">{selectedFile.name}</p>
                <p className="text-xs text-green-600">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <button
                onClick={() => setSelectedFile(null)}
                className="text-red-600 hover:text-red-800"
              >
                ×
              </button>
            </div>
          ) : (
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileInputChange}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Choose File
              </button>
            </div>
          )}
        </div>
      </motion.div>

      {/* Configuration Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg border border-gray-200 p-6"
      >
        <div className="flex items-center space-x-2 mb-4">
          <SparklesIcon className="h-5 w-5 text-purple-600" />
          <h3 className="text-lg font-semibold text-gray-900">Auto Configuration</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium text-gray-900">Time Budget</h4>
            <p className="text-sm text-gray-600">{formatTime(config.time_budget)}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium text-gray-900">Optimization</h4>
            <p className="text-sm text-gray-600">Auto-detect best metric</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-medium text-gray-900">Target Column</h4>
            <p className="text-sm text-gray-600">Auto-detect from data</p>
          </div>
        </div>
      </motion.div>

      {/* Start Pipeline Button */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <button
          onClick={handleStartPipeline}
          disabled={!selectedFile}
          className={`inline-flex items-center px-8 py-4 border border-transparent text-lg font-medium rounded-lg transition-colors ${
            selectedFile
              ? 'text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
              : 'text-gray-400 bg-gray-200 cursor-not-allowed'
          }`}
        >
          <PlayIcon className="h-6 w-6 mr-3" />
          {selectedFile ? 'Start ML Pipeline' : 'Upload Dataset First'}
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
            <h3 className="text-sm font-medium text-blue-800">How It Works</h3>
            <ul className="mt-1 text-sm text-blue-700 space-y-1">
              <li>• Upload your CSV dataset and we'll automatically detect the target column</li>
              <li>• AI will analyze your data and choose the best preprocessing steps</li>
              <li>• Multiple ML models will be trained and optimized automatically</li>
              <li>• Get detailed results with model performance and feature importance</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default PipelineConfigComponent;