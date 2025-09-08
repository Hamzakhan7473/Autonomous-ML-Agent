"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Navigation } from "@/components/layout/navigation"
import { Hero } from "@/components/landing/hero"
import { Header } from "@/components/layout/header"
import { Sidebar } from "@/components/layout/sidebar"
import { MobileStepIndicator } from "@/components/layout/mobile-step-indicator"
import { DataUpload } from "@/components/dashboard/data-upload"
import { PipelineConfig } from "@/components/dashboard/pipeline-config"
import { PipelineExecution } from "@/components/dashboard/pipeline-execution"
import { ResultsVisualization } from "@/components/dashboard/results-visualization"

export default function Home() {
  const [isDark, setIsDark] = useState(false)
  const [currentPage, setCurrentPage] = useState('home')
  const [currentStep, setCurrentStep] = useState(0)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  
  // Pipeline configuration state
  const [targetColumn, setTargetColumn] = useState("")
  const [optimizationMetric, setOptimizationMetric] = useState("accuracy")
  const [timeBudget, setTimeBudget] = useState(60)
  const [maxModels, setMaxModels] = useState(10)
  const [cvFolds, setCvFolds] = useState(5)
  const [enableEnsemble, setEnableEnsemble] = useState(true)
  const [enableInterpretability, setEnableInterpretability] = useState(true)
  const [enableMetaLearning, setEnableMetaLearning] = useState(true)
  
  // Pipeline execution state
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStepName, setCurrentStepName] = useState("")
  const [modelsTrained, setModelsTrained] = useState(0)
  const [elapsedTime, setElapsedTime] = useState(0)
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState(0)
  
  // Results state
  const [results, setResults] = useState(null)

  // Theme toggle
  const toggleTheme = () => {
    setIsDark(!isDark)
    document.documentElement.classList.toggle('dark')
  }

  // Navigation handler
  const handleNavigate = (page: string) => {
    setCurrentPage(page)
    if (page === 'dashboard') {
      setCurrentStep(0)
    }
  }

  // Get started handler
  const handleGetStarted = () => {
    setCurrentPage('dashboard')
    setCurrentStep(0)
  }

  // File upload handler
  const handleFileUpload = (file: File) => {
    setUploadedFile(file)
    setCurrentStep(1)
  }

  const handleRemoveFile = () => {
    setUploadedFile(null)
    setCurrentStep(0)
  }

  // Pipeline execution handlers
  const handleStartPipeline = async () => {
    if (!uploadedFile || !targetColumn) return
    
    setIsRunning(true)
    setCurrentStep(2)
    setProgress(0)
    setModelsTrained(0)
    setElapsedTime(0)
    
    // Simulate pipeline execution
    const totalSteps = 7
    const stepDuration = 3000 // 3 seconds per step
    
    for (let i = 0; i < totalSteps; i++) {
      setCurrentStepName([
        "Data Ingestion",
        "Data Preprocessing", 
        "Model Selection",
        "Hyperparameter Optimization",
        "Model Training",
        "Ensemble Building",
        "Model Interpretation"
      ][i])
      
      // Simulate model training progress
      const modelsPerStep = Math.floor(maxModels / totalSteps)
      for (let j = 0; j < modelsPerStep; j++) {
        await new Promise(resolve => setTimeout(resolve, stepDuration / modelsPerStep))
        setModelsTrained(prev => prev + 1)
        setProgress(prev => prev + (100 / maxModels))
      }
      
      setElapsedTime(prev => prev + stepDuration / 1000)
    }
    
    // Generate mock results
    setResults({
      bestModel: "XGBoost",
      leaderboard: [
        { name: "XGBoost", accuracy: 0.9234, precision: 0.9156, recall: 0.9289, f1: 0.9222, trainingTime: 45.2, crossValScore: 0.9187, crossValStd: 0.0123 },
        { name: "Random Forest", accuracy: 0.9102, precision: 0.9056, recall: 0.9123, f1: 0.9089, trainingTime: 32.1, crossValScore: 0.9078, crossValStd: 0.0156 },
        { name: "LightGBM", accuracy: 0.9087, precision: 0.9012, recall: 0.9145, f1: 0.9078, trainingTime: 28.9, crossValScore: 0.9056, crossValStd: 0.0134 },
        { name: "Logistic Regression", accuracy: 0.8756, precision: 0.8678, recall: 0.8823, f1: 0.8750, trainingTime: 12.3, crossValScore: 0.8723, crossValStd: 0.0189 },
        { name: "k-NN", accuracy: 0.8234, precision: 0.8156, recall: 0.8289, f1: 0.8222, trainingTime: 8.7, crossValScore: 0.8187, crossValStd: 0.0223 }
      ],
      featureImportance: {
        "feature_1": 0.234,
        "feature_2": 0.189,
        "feature_3": 0.156,
        "feature_4": 0.134,
        "feature_5": 0.112,
        "feature_6": 0.089,
        "feature_7": 0.067,
        "feature_8": 0.045,
        "feature_9": 0.034,
        "feature_10": 0.023
      },
      modelInsights: "The XGBoost model achieved the best performance with 92.34% accuracy. Key insights include: 1) Feature_1 and Feature_2 are the most important predictors, 2) The ensemble approach improved performance by 2.3% over individual models, 3) Cross-validation shows consistent performance across folds, 4) The model shows good generalization with low overfitting risk. Recommendations: Consider feature engineering for Feature_3 and Feature_4 to potentially improve performance further.",
      trainingTime: 180.5,
      totalIterations: maxModels
    })
    
    setIsRunning(false)
    setCurrentStep(3)
  }

  const handleStopPipeline = () => {
    setIsRunning(false)
    setProgress(0)
    setModelsTrained(0)
    setElapsedTime(0)
  }

  // Step content renderer
  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return (
          <DataUpload
            onFileUpload={handleFileUpload}
            uploadedFile={uploadedFile}
            onRemoveFile={handleRemoveFile}
          />
        )
      case 1:
        return (
          <PipelineConfig
            targetColumn={targetColumn}
            onTargetColumnChange={setTargetColumn}
            optimizationMetric={optimizationMetric}
            onOptimizationMetricChange={setOptimizationMetric}
            timeBudget={timeBudget}
            onTimeBudgetChange={setTimeBudget}
            maxModels={maxModels}
            onMaxModelsChange={setMaxModels}
            cvFolds={cvFolds}
            onCvFoldsChange={setCvFolds}
            enableEnsemble={enableEnsemble}
            onEnableEnsembleChange={setEnableEnsemble}
            enableInterpretability={enableInterpretability}
            onEnableInterpretabilityChange={setEnableInterpretability}
            enableMetaLearning={enableMetaLearning}
            onEnableMetaLearningChange={setEnableMetaLearning}
          />
        )
      case 2:
        return (
          <PipelineExecution
            onStartPipeline={handleStartPipeline}
            onStopPipeline={handleStopPipeline}
            isRunning={isRunning}
            progress={progress}
            currentStep={currentStepName}
            modelsTrained={modelsTrained}
            totalModels={maxModels}
            elapsedTime={elapsedTime}
            estimatedTimeRemaining={estimatedTimeRemaining}
          />
        )
      case 3:
        return <ResultsVisualization results={results} />
      default:
        return null
    }
  }

  // Render content based on current page
  const renderPageContent = () => {
    if (currentPage === 'home') {
      return (
        <div className="min-h-screen">
          <Hero onGetStarted={handleGetStarted} />
        </div>
      )
    }

    return (
      <div className="flex h-screen bg-background">
        {/* Sidebar */}
        <Sidebar currentStep={currentStep} onStepClick={setCurrentStep} />
        
        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <Header onToggleTheme={toggleTheme} isDark={isDark} />
          
          {/* Mobile Step Indicator */}
          <MobileStepIndicator currentStep={currentStep} onStepClick={setCurrentStep} />
          
          {/* Content Area */}
          <main className="flex-1 overflow-auto p-4 lg:p-6">
            <motion.div
              key={currentStep}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="max-w-6xl mx-auto"
            >
              {renderStepContent()}
            </motion.div>
          </main>
        </div>
      </div>
    )
  }

  return (
    <div className={`min-h-screen ${isDark ? 'dark' : ''}`}>
      {/* Navigation */}
      <Navigation 
        isDark={isDark}
        onToggleTheme={toggleTheme}
        onNavigate={handleNavigate}
        currentPage={currentPage}
      />
      
      {/* Page Content */}
      {renderPageContent()}
    </div>
  )
}