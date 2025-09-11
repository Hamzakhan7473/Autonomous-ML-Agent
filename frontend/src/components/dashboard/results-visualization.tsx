'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  TrophyIcon,
  DocumentArrowDownIcon,
  EyeIcon,
  MagnifyingGlassIcon,
  InformationCircleIcon,
  TableCellsIcon,
  ChartPieIcon
} from '@heroicons/react/24/outline';
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter
} from 'recharts';
import { useTaskMonitor, usePredictions } from '@/lib/hooks';
import { LeaderboardEntry, ModelResult, FeatureImportance } from '@/lib/types';

interface ResultsVisualizationProps {
  taskId?: string;
  className?: string;
}

const ResultsVisualizationComponent: React.FC<ResultsVisualizationProps> = ({ 
  taskId,
  className = '' 
}) => {
  const [activeTab, setActiveTab] = useState<'leaderboard' | 'performance' | 'features' | 'predictions'>('leaderboard');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');
  
  const { status } = useTaskMonitor(taskId || null);
  const { predictions, makePrediction } = usePredictions();

  // Mock data - in real implementation, this would come from the API
  const leaderboardData: LeaderboardEntry[] = [
    {
      rank: 1,
      model_name: 'XGBoost',
      score: 0.945,
      accuracy: 0.945,
      precision: 0.938,
      recall: 0.952,
      f1_score: 0.945,
      training_time: 45.2,
      parameters: { n_estimators: 100, max_depth: 6, learning_rate: 0.1 }
    },
    {
      rank: 2,
      model_name: 'Random Forest',
      score: 0.932,
      accuracy: 0.932,
      precision: 0.926,
      recall: 0.938,
      f1_score: 0.932,
      training_time: 32.1,
      parameters: { n_estimators: 200, max_depth: 10, min_samples_split: 2 }
    },
    {
      rank: 3,
      model_name: 'LightGBM',
      score: 0.928,
      accuracy: 0.928,
      precision: 0.921,
      recall: 0.935,
      f1_score: 0.928,
      training_time: 28.7,
      parameters: { n_estimators: 150, max_depth: 8, learning_rate: 0.05 }
    },
    {
      rank: 4,
      model_name: 'Logistic Regression',
      score: 0.891,
      accuracy: 0.891,
      precision: 0.885,
      recall: 0.897,
      f1_score: 0.891,
      training_time: 5.3,
      parameters: { C: 1.0, max_iter: 1000 }
    },
    {
      rank: 5,
      model_name: 'Neural Network',
      score: 0.876,
      accuracy: 0.876,
      precision: 0.869,
      recall: 0.883,
      f1_score: 0.876,
      training_time: 78.9,
      parameters: { hidden_layers: [100, 50], learning_rate: 0.001, epochs: 100 }
    }
  ];

  const featureImportance: FeatureImportance[] = [
    { feature: 'feature_1', importance: 0.245, relative_importance: 24.5 },
    { feature: 'feature_2', importance: 0.189, relative_importance: 18.9 },
    { feature: 'feature_3', importance: 0.156, relative_importance: 15.6 },
    { feature: 'feature_4', importance: 0.134, relative_importance: 13.4 },
    { feature: 'feature_5', importance: 0.098, relative_importance: 9.8 },
    { feature: 'feature_6', importance: 0.087, relative_importance: 8.7 },
    { feature: 'feature_7', importance: 0.065, relative_importance: 6.5 },
    { feature: 'feature_8', importance: 0.026, relative_importance: 2.6 }
  ];

  const performanceMetrics = leaderboardData.map(model => ({
    name: model.model_name,
    accuracy: model.accuracy!,
    precision: model.precision!,
    recall: model.recall!,
    f1_score: model.f1_score!,
    training_time: model.training_time
  }));

  const tabs = [
    { id: 'leaderboard', name: 'Leaderboard', icon: TrophyIcon },
    { id: 'performance', name: 'Performance', icon: ChartBarIcon },
    { id: 'features', name: 'Features', icon: MagnifyingGlassIcon },
    { id: 'predictions', name: 'Predictions', icon: EyeIcon },
  ];

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  const renderLeaderboard = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900">Model Leaderboard</h3>
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode('chart')}
            className={`px-3 py-1 text-sm rounded-md ${
              viewMode === 'chart' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Chart
          </button>
          <button
            onClick={() => setViewMode('table')}
            className={`px-3 py-1 text-sm rounded-md ${
              viewMode === 'table' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Table
          </button>
        </div>
      </div>

      {viewMode === 'chart' ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h4 className="text-md font-semibold text-gray-900 mb-4">Model Performance Comparison</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={performanceMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="accuracy" fill="#3B82F6" name="Accuracy" />
                <Bar dataKey="precision" fill="#10B981" name="Precision" />
                <Bar dataKey="recall" fill="#F59E0B" name="Recall" />
                <Bar dataKey="f1_score" fill="#EF4444" name="F1 Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h4 className="text-md font-semibold text-gray-900 mb-4">Training Time vs Performance</h4>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart data={performanceMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="training_time" name="Training Time (s)" />
                <YAxis dataKey="accuracy" name="Accuracy" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter dataKey="accuracy" fill="#3B82F6" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Precision</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Recall</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">F1 Score</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time (s)</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {leaderboardData.map((model) => (
                <tr key={model.model_name} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      {model.rank === 1 && <TrophyIcon className="h-5 w-5 text-yellow-500 mr-2" />}
                      <span className="text-sm font-medium text-gray-900">{model.rank}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {model.model_name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {(model.score * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {((model.accuracy || 0) * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {((model.precision || 0) * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {((model.recall || 0) * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {((model.f1_score || 0) * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {model.training_time.toFixed(1)}s
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  const renderPerformance = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900">Performance Analysis</h3>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">Metric Trends</h4>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceMetrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="accuracy" stroke="#3B82F6" name="Accuracy" />
              <Line type="monotone" dataKey="precision" stroke="#10B981" name="Precision" />
              <Line type="monotone" dataKey="recall" stroke="#F59E0B" name="Recall" />
              <Line type="monotone" dataKey="f1_score" stroke="#EF4444" name="F1 Score" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">Best Model Distribution</h4>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={[
                  { name: 'XGBoost', value: 45 },
                  { name: 'Random Forest', value: 25 },
                  { name: 'LightGBM', value: 20 },
                  { name: 'Others', value: 10 }
                ]}
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }: any) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {[0, 1, 2, 3].map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  const renderFeatures = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900">Feature Importance</h3>
      
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h4 className="text-md font-semibold text-gray-900 mb-4">Top Features by Importance</h4>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={featureImportance} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="feature" type="category" width={120} />
            <Tooltip formatter={(value) => [`${value}%`, 'Importance']} />
            <Bar dataKey="relative_importance" fill="#3B82F6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">Feature Statistics</h4>
          <div className="space-y-3">
            {featureImportance.slice(0, 5).map((feature, index) => (
              <div key={feature.feature} className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700">{feature.feature}</span>
                <span className="text-sm text-gray-900">{feature.relative_importance.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h4 className="text-md font-semibold text-gray-900 mb-4">Feature Insights</h4>
          <ul className="space-y-2 text-sm text-gray-600">
            <li>• Top 3 features account for 58.8% of importance</li>
            <li>• Feature_1 is the most predictive variable</li>
            <li>• Last 3 features contribute less than 20%</li>
            <li>• Consider feature engineering for low-importance features</li>
          </ul>
        </div>
      </div>
    </div>
  );

  const renderPredictions = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900">Model Predictions</h3>
        <button className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 rounded-md hover:bg-blue-100">
          <DocumentArrowDownIcon className="h-4 w-4" />
          <span>Export Predictions</span>
        </button>
      </div>

      {predictions.length === 0 ? (
        <div className="text-center py-8 bg-white rounded-lg border border-gray-200">
          <EyeIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No predictions yet</h3>
          <p className="mt-1 text-sm text-gray-500">
            Use the prediction interface to make predictions with your trained model.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {predictions.map((prediction, index) => (
            <div key={prediction.id} className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h4 className="text-md font-semibold text-gray-900">Prediction #{index + 1}</h4>
                  <p className="text-sm text-gray-600">
                    {prediction.timestamp.toLocaleString()}
                  </p>
                </div>
                {prediction.confidence && (
                  <span className="px-2 py-1 text-xs font-medium text-green-800 bg-green-100 rounded-full">
                    Confidence: {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h5 className="text-sm font-medium text-gray-700 mb-2">Input Features</h5>
                  <div className="space-y-1">
                    {prediction.features[0]?.map((value: any, i: number) => (
                      <div key={i} className="flex justify-between text-sm">
                        <span className="text-gray-600">
                          {prediction.featureNames?.[i] || `Feature ${i + 1}`}:
                        </span>
                        <span className="text-gray-900">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h5 className="text-sm font-medium text-gray-700 mb-2">Predictions</h5>
                  <div className="space-y-2">
                    {prediction.predictions.map((pred: any, i: number) => (
                      <div key={i} className="flex justify-between text-sm">
                        <span className="text-gray-600">Prediction {i + 1}:</span>
                        <span className="font-medium text-gray-900">{pred}</span>
                      </div>
                    ))}
                  </div>

                  {prediction.probabilities && (
                    <div className="mt-4">
                      <h6 className="text-sm font-medium text-gray-700 mb-2">Probabilities</h6>
                      <div className="space-y-1">
                        {prediction.probabilities[0]?.map((prob: any, i: number) => (
                          <div key={i} className="flex justify-between text-sm">
                            <span className="text-gray-600">Class {i}:</span>
                            <span className="text-gray-900">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'leaderboard':
        return renderLeaderboard();
      case 'performance':
        return renderPerformance();
      case 'features':
        return renderFeatures();
      case 'predictions':
        return renderPredictions();
      default:
        return renderLeaderboard();
    }
  };

  if (!taskId || !status || status.status !== 'completed') {
    return (
      <div className={`text-center py-12 ${className}`}>
        <ChartBarIcon className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">No Results Available</h3>
        <p className="mt-1 text-sm text-gray-500">
          Complete a pipeline execution to view results and analysis.
        </p>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900">Results & Analysis</h2>
        <p className="mt-2 text-gray-600">
          Comprehensive analysis of your machine learning pipeline results
        </p>
      </div>

      {/* Navigation */}
      <div className="bg-white rounded-lg border border-gray-200 p-1">
        <nav className="flex space-x-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.name}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {renderActiveTab()}
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
            <h3 className="text-sm font-medium text-blue-800">Results Information</h3>
            <ul className="mt-1 text-sm text-blue-700 space-y-1">
              <li>• Best performing model: {leaderboardData[0].model_name} ({(leaderboardData[0].score * 100).toFixed(1)}%)</li>
              <li>• Total models evaluated: {leaderboardData.length}</li>
              <li>• Average training time: {(leaderboardData.reduce((sum, m) => sum + m.training_time, 0) / leaderboardData.length).toFixed(1)}s</li>
              <li>• Results saved to: {status.results?.output_dir || './results'}</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default ResultsVisualizationComponent;