'use client';

import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  CloudArrowUpIcon, 
  DocumentIcon, 
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import { useDatasetUpload } from '@/lib/hooks';
import { DatasetInfo } from '@/lib/types';
import { formatFileSize } from '@/lib/api';

interface DatasetUploadProps {
  onUploadSuccess?: (datasetInfo: DatasetInfo) => void;
  className?: string;
}

const DatasetUpload: React.FC<DatasetUploadProps> = ({ 
  onUploadSuccess, 
  className = '' 
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const { uploading, analyzing, datasetInfo, error, uploadAndAnalyze, reset } = useDatasetUpload();

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (isValidFile(file)) {
        setSelectedFile(file);
        uploadAndAnalyze(file);
      }
    }
  }, [uploadAndAnalyze]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (isValidFile(file)) {
        setSelectedFile(file);
        uploadAndAnalyze(file);
      }
    }
  }, [uploadAndAnalyze]);

  const isValidFile = (file: File): boolean => {
    const validTypes = [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/json',
      'text/plain'
    ];
    
    const validExtensions = ['.csv', '.xlsx', '.xls', '.json', '.txt'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    return validTypes.includes(file.type) || validExtensions.includes(fileExtension);
  };

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    reset();
  }, [reset]);

  React.useEffect(() => {
    if (datasetInfo && onUploadSuccess) {
      onUploadSuccess(datasetInfo);
    }
  }, [datasetInfo, onUploadSuccess]);

  const renderUploadArea = () => (
    <div
      className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        dragActive
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-300 hover:border-gray-400'
      }`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        type="file"
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        onChange={handleFileSelect}
        accept=".csv,.xlsx,.xls,.json,.txt"
        disabled={uploading || analyzing}
      />
      
      <div className="space-y-4">
        <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
        <div>
          <p className="text-lg font-medium text-gray-900">
            {dragActive ? 'Drop your dataset here' : 'Upload your dataset'}
          </p>
          <p className="text-sm text-gray-600">
            Drag and drop or click to browse
          </p>
        </div>
        <div className="text-xs text-gray-500">
          Supports CSV, Excel, JSON files up to 100MB
        </div>
      </div>
    </div>
  );

  const renderFileInfo = () => {
    if (!selectedFile) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg border border-gray-200 p-6"
      >
        <div className="flex items-center space-x-4">
          <DocumentIcon className="h-8 w-8 text-blue-500" />
          <div className="flex-1">
            <h3 className="text-lg font-medium text-gray-900">{selectedFile.name}</h3>
            <p className="text-sm text-gray-600">
              {formatFileSize(selectedFile.size)} • {selectedFile.type || 'Unknown type'}
            </p>
          </div>
          <button
            onClick={handleReset}
            className="text-sm text-gray-500 hover:text-gray-700"
            disabled={uploading || analyzing}
          >
            Remove
          </button>
        </div>

        {(uploading || analyzing) && (
          <div className="mt-4">
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
              <span className="text-sm text-gray-600">
                {uploading ? 'Uploading...' : 'Analyzing dataset...'}
              </span>
            </div>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
            </div>
          </div>
        )}
      </motion.div>
    );
  };

  const renderDatasetInfo = () => {
    if (!datasetInfo) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <CheckCircleIcon className="h-6 w-6 text-green-500" />
            <h3 className="text-lg font-semibold text-gray-900">Dataset Analysis Complete</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">Dataset Shape</h4>
              <p className="text-2xl font-bold text-gray-900">
                {datasetInfo.shape[0].toLocaleString()} × {datasetInfo.shape[1]}
              </p>
              <p className="text-sm text-gray-600">rows × columns</p>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">Target Column</h4>
              <p className="text-lg font-semibold text-gray-900">{datasetInfo.target_column}</p>
              <p className="text-sm text-gray-600 capitalize">{datasetInfo.target_type}</p>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">Missing Values</h4>
              <p className="text-2xl font-bold text-gray-900">
                {(datasetInfo.missing_percentage * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-gray-600">of total data</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Column Information</h4>
          <div className="space-y-2">
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-sm font-medium text-gray-700">Total Columns:</span>
              <span className="text-sm text-gray-900">{datasetInfo.columns.length}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-gray-100">
              <span className="text-sm font-medium text-gray-700">Features:</span>
              <span className="text-sm text-gray-900">
                {datasetInfo.columns.length - 1}
              </span>
            </div>
            <div className="flex justify-between items-center py-2">
              <span className="text-sm font-medium text-gray-700">Target:</span>
              <span className="text-sm text-gray-900">1</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Column Names</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
            {datasetInfo.columns.map((column, index) => (
              <div
                key={index}
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  column === datasetInfo.target_column
                    ? 'bg-blue-100 text-blue-800'
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                {column}
                {column === datasetInfo.target_column && (
                  <span className="ml-1 text-xs">(target)</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    );
  };

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
          <h3 className="text-sm font-medium text-red-800">Upload Failed</h3>
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
        <h2 className="text-2xl font-bold text-gray-900">Upload Dataset</h2>
        <p className="mt-2 text-gray-600">
          Upload your dataset to start the autonomous ML pipeline
        </p>
      </div>

      {/* Upload Area */}
      {!selectedFile && !datasetInfo && !error && renderUploadArea()}

      {/* File Info */}
      {selectedFile && renderFileInfo()}

      {/* Error */}
      {error && renderError()}

      {/* Dataset Analysis Results */}
      {datasetInfo && renderDatasetInfo()}

      {/* Info Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-blue-50 border border-blue-200 rounded-lg p-4"
      >
        <div className="flex items-start space-x-2">
          <InformationCircleIcon className="h-5 w-5 text-blue-500 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-blue-800">Dataset Requirements</h3>
            <ul className="mt-1 text-sm text-blue-700 space-y-1">
              <li>• Supported formats: CSV, Excel (.xlsx, .xls), JSON</li>
              <li>• Maximum file size: 100MB</li>
              <li>• Target column should be clearly identifiable</li>
              <li>• Missing values will be automatically handled</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default DatasetUpload;
