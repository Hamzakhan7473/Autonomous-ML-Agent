'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  CpuChipIcon, 
  ChartBarIcon, 
  Cog6ToothIcon, 
  PlayIcon,
  DocumentTextIcon,
  CloudArrowUpIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';
import { usePipelineTasks, useSystemHealth } from '@/lib/hooks';
import { DashboardStats } from '@/lib/types';
import { formatDuration } from '@/lib/api';
import DatasetUpload from './dataset-upload';
import PipelineConfig from './pipeline-config';
import PipelineExecution from './pipeline-execution';
import ResultsVisualization from './results-visualization';

interface DashboardProps {
  className?: string;
}

const Dashboard: React.FC<DashboardProps> = ({ className = '' }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'upload' | 'config' | 'execute' | 'results'>('overview');
  const { tasks, loading: tasksLoading } = usePipelineTasks();
  const { health, loading: healthLoading } = useSystemHealth();

  // Calculate dashboard statistics
  const stats: DashboardStats = {
    total_tasks: tasks.length,
    completed_tasks: tasks.filter(t => t.status === 'completed').length,
    failed_tasks: tasks.filter(t => t.status === 'failed').length,
    active_tasks: tasks.filter(t => t.status === 'running').length,
    average_training_time: tasks.length > 0 
      ? tasks.reduce((sum, t) => sum + (t.estimatedTime || 0), 0) / tasks.length 
      : 0,
    best_model_score: Math.max(...tasks.map(t => t.results?.best_score || 0), 0),
    total_models_trained: tasks.reduce((sum, t) => sum + (t.results?.model_count || 0), 0)
  };

  const tabs = [
    { id: 'overview', name: 'Overview', icon: ChartBarIcon },
    { id: 'upload', name: 'Upload Data', icon: CloudArrowUpIcon },
    { id: 'config', name: 'Configure', icon: Cog6ToothIcon },
    { id: 'execute', name: 'Execute', icon: PlayIcon },
    { id: 'results', name: 'Results', icon: DocumentTextIcon },
  ];

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    icon: React.ComponentType<any>;
    color: string;
    trend?: number;
  }> = ({ title, value, icon: Icon, color, trend }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
        <div className={`p-3 rounded-full ${color}`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
      </div>
      {trend !== undefined && (
        <div className="mt-2">
          <span className={`text-sm ${trend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {trend >= 0 ? '↗' : '↘'} {Math.abs(trend)}%
          </span>
        </div>
      )}
    </motion.div>
  );

  const renderOverview = () => (
    <div className="space-y-6">
      {/* System Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              health?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
            }`} />
            <span className="text-sm font-medium text-gray-700">
              {health?.status === 'healthy' ? 'System Healthy' : 'System Issues'}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <CpuChipIcon className="h-4 w-4 text-gray-400" />
            <span className="text-sm text-gray-600">
              LLM Client: {health?.llm_client_available ? 'Available' : 'Unavailable'}
            </span>
          </div>
        </div>
      </motion.div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Tasks"
          value={stats.total_tasks}
          icon={DocumentTextIcon}
          color="bg-blue-500"
        />
        <StatCard
          title="Completed"
          value={stats.completed_tasks}
          icon={CheckCircleIcon}
          color="bg-green-500"
        />
        <StatCard
          title="Active"
          value={stats.active_tasks}
          icon={ClockIcon}
          color="bg-yellow-500"
        />
        <StatCard
          title="Failed"
          value={stats.failed_tasks}
          icon={XCircleIcon}
          color="bg-red-500"
        />
      </div>

      {/* Recent Tasks */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Tasks</h3>
        {tasks.length === 0 ? (
          <div className="text-center py-8">
            <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No tasks yet</h3>
            <p className="mt-1 text-sm text-gray-500">
              Upload a dataset and start your first ML pipeline.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {tasks.slice(0, 5).map((task) => (
              <div key={task.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${
                    task.status === 'completed' ? 'bg-green-500' :
                    task.status === 'running' ? 'bg-blue-500' :
                    task.status === 'failed' ? 'bg-red-500' : 'bg-gray-400'
                  }`} />
                  <div>
                    <p className="text-sm font-medium text-gray-900">{task.dataset.name}</p>
                    <p className="text-xs text-gray-500">
                      {task.startTime.toLocaleDateString()} at {task.startTime.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium text-gray-900 capitalize">{task.status}</p>
                  {task.status === 'running' && (
                    <p className="text-xs text-gray-500">
                      {Math.round(task.progress * 100)}% complete
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverview();
      case 'upload':
        return <DatasetUpload onUploadSuccess={() => setActiveTab('config')} />;
      case 'config':
        return <PipelineConfig onConfigReady={() => setActiveTab('execute')} />;
      case 'execute':
        return <PipelineExecution onExecutionComplete={() => setActiveTab('results')} />;
      case 'results':
        return <ResultsVisualization />;
      default:
        return renderOverview();
    }
  };

  return (
    <div className={`min-h-screen bg-gray-50 ${className}`}>
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Autonomous ML Agent</h1>
              <p className="text-sm text-gray-600">AI-powered machine learning pipeline orchestration</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                health?.status === 'healthy' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {healthLoading ? 'Checking...' : health?.status === 'healthy' ? 'Online' : 'Offline'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderActiveTab()}
      </div>
    </div>
  );
};

export default Dashboard;
