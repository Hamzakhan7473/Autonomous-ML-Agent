'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CpuChipIcon, 
  ChartBarIcon, 
  Cog6ToothIcon, 
  PlayIcon,
  DocumentTextIcon,
  CloudArrowUpIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  SparklesIcon,
  BeakerIcon,
  RocketLaunchIcon,
  LightBulbIcon,
  ShieldCheckIcon,
  ArrowPathIcon,
  EyeIcon,
  DocumentArrowDownIcon,
  CogIcon,
  ChartPieIcon
} from '@heroicons/react/24/outline';
import { usePipelineTasks, useSystemHealth } from '@/lib/hooks';
import { DashboardStats } from '@/lib/types';
import { formatDuration } from '@/lib/api';
import DatasetUpload from './dataset-upload';
import PipelineConfig from './pipeline-config';
import PipelineExecution from './pipeline-execution';
import ResultsVisualization from './results-visualization';
import Logo from '../ui/logo';

interface DashboardProps {
  className?: string;
}

const Dashboard: React.FC<DashboardProps> = ({ className = '' }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'upload' | 'execute' | 'results' | 'insights' | 'export'>('overview');
  const { tasks, loading: tasksLoading } = usePipelineTasks();
  const { health, loading: healthLoading } = useSystemHealth();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

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
    { id: 'overview', name: 'Dashboard', icon: ChartBarIcon, description: 'System overview and quick stats' },
    { id: 'upload', name: 'Start Pipeline', icon: RocketLaunchIcon, description: 'Upload data and configure ML pipeline' },
    { id: 'execute', name: 'Monitor', icon: PlayIcon, description: 'Track pipeline execution progress' },
    { id: 'results', name: 'Results', icon: ChartPieIcon, description: 'View model performance and leaderboard' },
    { id: 'insights', name: 'AI Insights', icon: LightBulbIcon, description: 'Natural language analysis and recommendations' },
    { id: 'export', name: 'Export', icon: DocumentArrowDownIcon, description: 'Download models and deployment artifacts' },
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
      whileHover={{ scale: 1.02, boxShadow: "0 10px 25px rgba(0,0,0,0.3)" }}
      className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6 hover:bg-white/10 transition-all duration-300"
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
        </div>
        <div className={`p-3 rounded-full bg-gradient-to-r ${color} shadow-lg`}>
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
      {/* Welcome Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-2xl p-8 border border-white/10 backdrop-blur-md"
      >
        <div className="flex items-center space-x-4">
          <Logo size="lg" animated={true} />
          <div>
            <h1 className="text-3xl font-bold text-white">Welcome to Autonomous ML</h1>
            <p className="text-lg text-gray-300 mt-2">
              AI-powered machine learning pipeline orchestration with advanced features
            </p>
            <div className="flex items-center space-x-4 mt-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setActiveTab('upload')}
                className="bg-gradient-to-r from-cyan-600 to-blue-600 text-white px-6 py-3 rounded-full font-medium hover:from-cyan-700 hover:to-blue-700 transition-all duration-300 flex items-center space-x-2 shadow-lg"
              >
                <RocketLaunchIcon className="h-5 w-5" />
                <span>Start New Pipeline</span>
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setActiveTab('results')}
                className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-6 py-3 rounded-full font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-300 flex items-center space-x-2 shadow-lg"
              >
                <ChartPieIcon className="h-5 w-5" />
                <span>View Results</span>
              </motion.button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* System Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              className={`w-3 h-3 rounded-full ${
                health?.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'
              }`}
            />
            <span className="text-sm font-medium text-gray-300">
              {health?.status === 'healthy' ? 'System Operational' : 'System Issues'}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <CpuChipIcon className="h-4 w-4 text-cyan-400" />
            <span className="text-sm text-gray-300">
              E2B Client: {health?.llm_client_available ? 'Active' : 'Inactive'}
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
        className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white">Recent Tasks</h3>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setActiveTab('execute')}
            className="text-sm text-cyan-400 hover:text-cyan-300 transition-colors"
          >
            View All →
          </motion.button>
        </div>
        
        {tasks.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-12"
          >
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/20 to-purple-400/20 rounded-full blur-2xl"></div>
              <DocumentTextIcon className="relative mx-auto h-16 w-16 text-cyan-400 mb-4" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">No tasks yet</h3>
            <p className="text-gray-400 mb-6 max-w-md mx-auto">
              Upload a dataset and start your first ML pipeline to see amazing results.
            </p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setActiveTab('upload')}
              className="bg-gradient-to-r from-cyan-600 to-blue-600 text-white px-6 py-3 rounded-full font-medium hover:from-cyan-700 hover:to-blue-700 transition-all duration-300 shadow-lg"
            >
              Get Started
            </motion.button>
          </motion.div>
        ) : (
          <div className="space-y-3">
            {tasks.slice(0, 5).map((task, index) => (
              <motion.div
                key={task.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.02, backgroundColor: "rgba(255,255,255,0.1)" }}
                className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/10 hover:border-white/20 transition-all duration-300"
              >
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 2, repeat: Infinity, delay: index * 0.2 }}
                      className={`w-3 h-3 rounded-full ${
                        task.status === 'completed' ? 'bg-green-400' :
                        task.status === 'running' ? 'bg-cyan-400' :
                        task.status === 'failed' ? 'bg-red-400' : 'bg-gray-400'
                      }`}
                    />
                    {task.status === 'running' && (
                      <div className="absolute inset-0 w-3 h-3 rounded-full bg-cyan-400 animate-ping"></div>
                    )}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white">{task.dataset.name}</p>
                    <p className="text-xs text-gray-400">
                      {task.startTime.toLocaleDateString()} at {task.startTime.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                    task.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                    task.status === 'running' ? 'bg-cyan-500/20 text-cyan-400' :
                    task.status === 'failed' ? 'bg-red-500/20 text-red-400' : 'bg-gray-500/20 text-gray-400'
                  }`}>
                    {task.status}
                  </div>
                  {task.status === 'running' && (
                    <p className="text-xs text-gray-400 mt-1">
                      {Math.round(task.progress * 100)}% complete
                    </p>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );

  const handleStartPipeline = (file: File, config: any) => {
    setSelectedFile(file);
    setActiveTab('execute');
  };

  const renderActiveTab = () => {
    const completedTasks = tasks.filter(t => t.status === 'completed');
    
    const tabContent = {
      overview: renderOverview(),
      upload: <PipelineConfig onStartPipeline={handleStartPipeline} />,
      execute: <PipelineExecution 
        selectedFile={selectedFile}
        onExecutionComplete={() => setActiveTab('results')} 
      />,
      results: <ResultsVisualization completedTasks={completedTasks} />,
      insights: renderAIInsights(completedTasks),
      export: renderExportSection(completedTasks)
    };

    return tabContent[activeTab] || tabContent.overview;
  };

  const renderAIInsights = (completedTasks: any[]) => {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="space-y-6"
      >
        <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-2xl p-8 border border-white/10 backdrop-blur-md">
          <div className="flex items-center space-x-4 mb-6">
            <Logo size="lg" animated={true} />
            <div>
              <h2 className="text-3xl font-bold text-white">AI-Powered Insights</h2>
              <p className="text-gray-300 text-lg">Natural language analysis and intelligent recommendations</p>
            </div>
          </div>
        </div>

        {completedTasks.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-12 bg-white rounded-lg border border-gray-200"
          >
            <SparklesIcon className="mx-auto h-16 w-16 text-purple-400 mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No Insights Available Yet</h3>
            <p className="text-gray-600 mb-6">Complete a pipeline execution to unlock AI-powered insights and recommendations.</p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setActiveTab('upload')}
              className="bg-purple-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-purple-700 transition-colors"
            >
              Start Your First Pipeline
            </motion.button>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Executive Summary */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              whileHover={{ scale: 1.02, boxShadow: "0 20px 40px rgba(0,0,0,0.3)" }}
              className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl">
                  <DocumentTextIcon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">Executive Summary</h3>
              </div>
              <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 rounded-xl p-4 border border-blue-500/20">
                <p className="text-gray-300 leading-relaxed">
                  Your ML pipeline has successfully completed with impressive results. The system automatically 
                  evaluated multiple algorithms and selected the optimal model based on performance metrics.
                </p>
              </div>
            </motion.div>

            {/* Technical Analysis */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              whileHover={{ scale: 1.02, boxShadow: "0 20px 40px rgba(0,0,0,0.3)" }}
              className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-gradient-to-r from-green-600 to-emerald-600 rounded-xl">
                  <CogIcon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">Technical Analysis</h3>
              </div>
              <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 rounded-xl p-4 border border-green-500/20">
                <p className="text-gray-300 leading-relaxed">
                  The ensemble approach shows significant improvement over individual models. Feature importance 
                  analysis reveals key predictors that drive model performance.
                </p>
              </div>
            </motion.div>

            {/* Feature Insights */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              whileHover={{ scale: 1.02, boxShadow: "0 20px 40px rgba(0,0,0,0.3)" }}
              className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-gradient-to-r from-orange-600 to-amber-600 rounded-xl">
                  <BeakerIcon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">Feature Engineering Insights</h3>
              </div>
              <div className="bg-gradient-to-r from-orange-500/10 to-amber-500/10 rounded-xl p-4 border border-orange-500/20">
                <p className="text-gray-300 leading-relaxed">
                  Consider creating interaction features between the top predictors. Feature scaling 
                  optimization could further improve model stability and performance.
                </p>
              </div>
            </motion.div>

            {/* Deployment Recommendations */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              whileHover={{ scale: 1.02, boxShadow: "0 20px 40px rgba(0,0,0,0.3)" }}
              className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-gradient-to-r from-red-600 to-pink-600 rounded-xl">
                  <ShieldCheckIcon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">Deployment Recommendations</h3>
              </div>
              <div className="bg-gradient-to-r from-red-500/10 to-pink-500/10 rounded-xl p-4 border border-red-500/20">
                <p className="text-gray-300 leading-relaxed">
                  Implement model monitoring for performance tracking. Set up automated retraining 
                  pipelines and consider A/B testing for model comparison.
                </p>
              </div>
            </motion.div>
          </div>
        )}
      </motion.div>
    );
  };

  const renderExportSection = (completedTasks: any[]) => {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="space-y-6"
      >
        <div className="bg-gradient-to-r from-green-900/30 to-blue-900/30 rounded-2xl p-8 border border-white/10 backdrop-blur-md">
          <div className="flex items-center space-x-4 mb-6">
            <Logo size="lg" animated={true} />
            <div>
              <h2 className="text-3xl font-bold text-white">Export & Deploy</h2>
              <p className="text-gray-300 text-lg">Download models and deployment-ready artifacts</p>
            </div>
          </div>
        </div>

        {completedTasks.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-12 bg-white rounded-lg border border-gray-200"
          >
            <DocumentArrowDownIcon className="mx-auto h-16 w-16 text-green-400 mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No Models to Export</h3>
            <p className="text-gray-600 mb-6">Complete a pipeline execution to export your trained models and deployment artifacts.</p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setActiveTab('upload')}
              className="bg-green-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-green-700 transition-colors"
            >
              Create Your First Model
            </motion.button>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* FastAPI Service */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              whileHover={{ scale: 1.02, boxShadow: "0 20px 40px rgba(0,0,0,0.3)" }}
              className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl">
                  <RocketLaunchIcon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">FastAPI Service</h3>
              </div>
              <p className="text-gray-300 mb-6">
                Production-ready REST API with health checks, predictions, and batch processing.
              </p>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 text-white py-3 px-4 rounded-xl font-medium hover:from-blue-700 hover:to-cyan-700 transition-all duration-300 shadow-lg"
              >
                Download FastAPI Package
              </motion.button>
            </motion.div>

            {/* Python Package */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              whileHover={{ scale: 1.02, boxShadow: "0 20px 40px rgba(0,0,0,0.3)" }}
              className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl">
                  <Cog6ToothIcon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">Python Package</h3>
              </div>
              <p className="text-gray-300 mb-6">
                Easy-to-use Python package with predictor class for seamless integration.
              </p>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 px-4 rounded-xl font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-300 shadow-lg"
              >
                Download Python Package
              </motion.button>
            </motion.div>

            {/* Docker Container */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              whileHover={{ scale: 1.02, boxShadow: "0 20px 40px rgba(0,0,0,0.3)" }}
              className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-6 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-gradient-to-r from-green-600 to-emerald-600 rounded-xl">
                  <EyeIcon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">Docker Container</h3>
              </div>
              <p className="text-gray-300 mb-6">
                Containerized service with docker-compose for easy deployment.
              </p>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full bg-gradient-to-r from-green-600 to-emerald-600 text-white py-3 px-4 rounded-xl font-medium hover:from-green-700 hover:to-emerald-700 transition-all duration-300 shadow-lg"
              >
                Download Docker Package
              </motion.button>
            </motion.div>
          </div>
        )}
      </motion.div>
    );
  };

  return (
    <div className={`min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 ${className}`}>
      {/* Hero Header */}
      <div className="relative overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-cyan-600/20 animate-pulse"></div>
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <div className="mb-8 flex justify-center">
              <Logo size="xl" animated={true} />
            </div>
            
            <h1 className="text-6xl font-bold text-white mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Autonomous ML Agent
            </h1>
            <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed">
              Revolutionary AI-powered machine learning platform that orchestrates end-to-end pipelines with 
              <span className="text-cyan-400 font-semibold"> neural network intelligence</span> and 
              <span className="text-purple-400 font-semibold"> decentralized computing power</span>
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6">
              <motion.button
                whileHover={{ scale: 1.05, boxShadow: "0 20px 40px rgba(139, 92, 246, 0.3)" }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setActiveTab('upload')}
                className="group relative px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full font-semibold text-white shadow-2xl hover:shadow-purple-500/25 transition-all duration-300"
              >
                <span className="relative z-10 flex items-center space-x-2">
                  <RocketLaunchIcon className="h-5 w-5" />
                  <span>Launch Pipeline</span>
                </span>
                <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full blur opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </motion.button>
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setActiveTab('results')}
                className="px-8 py-4 border-2 border-white/20 rounded-full font-semibold text-white hover:bg-white/10 transition-all duration-300 backdrop-blur-sm"
              >
                <span className="flex items-center space-x-2">
                  <ChartPieIcon className="h-5 w-5" />
                  <span>View Results</span>
                </span>
              </motion.button>
            </div>
            
            {/* System Status */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-12 flex items-center justify-center space-x-8"
            >
              <div className="flex items-center space-x-2">
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className={`w-3 h-3 rounded-full ${health?.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'}`}
                />
                <span className="text-sm text-gray-300">
                  {health?.status === 'healthy' ? 'System Operational' : 'System Issues'}
                </span>
              </div>
              <div className="text-sm text-gray-400">•</div>
              <div className="text-sm text-gray-300">E2B Neural Network Active</div>
              <div className="text-sm text-gray-400">•</div>
              <div className="text-sm text-gray-300">12+ ML Algorithms Ready</div>
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* Navigation */}
      <div className="bg-black/20 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <motion.button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`relative flex flex-col items-center space-y-1 py-4 px-3 border-b-2 font-medium text-sm rounded-t-lg transition-all duration-200 ${
                    isActive
                      ? 'border-cyan-400 text-cyan-400 bg-cyan-500/10'
                      : 'border-transparent text-gray-400 hover:text-white hover:border-white/30 hover:bg-white/5'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <Icon className={`h-5 w-5 transition-colors duration-200 ${
                      isActive ? 'text-cyan-400' : 'text-gray-400'
                    }`} />
                    <span className="font-semibold">{tab.name}</span>
                  </div>
                  <motion.span 
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: isActive ? 1 : 0, height: isActive ? 'auto' : 0 }}
                    className="text-xs text-gray-500 text-center leading-tight"
                  >
                    {tab.description}
                  </motion.span>
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-400 to-purple-400"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.3 }}
                    />
                  )}
                </motion.button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 bg-gradient-to-b from-transparent to-black/20">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
          >
            {renderActiveTab()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Dashboard;
