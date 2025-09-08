"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { 
  PlayIcon,
  PauseIcon,
  StopIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  CpuChipIcon,
  ChartBarIcon,
  SparklesIcon
} from "@heroicons/react/24/outline"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface PipelineExecutionProps {
  onStartPipeline: () => void
  onStopPipeline: () => void
  isRunning: boolean
  progress: number
  currentStep: string
  modelsTrained: number
  totalModels: number
  elapsedTime: number
  estimatedTimeRemaining: number
}

interface PipelineStep {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'error'
  icon: any
  description: string
}

const pipelineSteps: PipelineStep[] = [
  {
    id: 'data-ingestion',
    name: 'Data Ingestion',
    status: 'pending',
    icon: ChartBarIcon,
    description: 'Loading and analyzing dataset'
  },
  {
    id: 'preprocessing',
    name: 'Data Preprocessing',
    status: 'pending',
    icon: CpuChipIcon,
    description: 'Cleaning and feature engineering'
  },
  {
    id: 'model-selection',
    name: 'Model Selection',
    status: 'pending',
    icon: SparklesIcon,
    description: 'AI-guided algorithm selection'
  },
  {
    id: 'hyperparameter-optimization',
    name: 'Hyperparameter Optimization',
    status: 'pending',
    icon: CpuChipIcon,
    description: 'Optimizing model parameters'
  },
  {
    id: 'model-training',
    name: 'Model Training',
    status: 'pending',
    icon: ChartBarIcon,
    description: 'Training multiple models'
  },
  {
    id: 'ensemble-building',
    name: 'Ensemble Building',
    status: 'pending',
    icon: SparklesIcon,
    description: 'Creating ensemble models'
  },
  {
    id: 'interpretation',
    name: 'Model Interpretation',
    status: 'pending',
    icon: ChartBarIcon,
    description: 'Generating insights and explanations'
  }
]

export function PipelineExecution({
  onStartPipeline,
  onStopPipeline,
  isRunning,
  progress,
  currentStep,
  modelsTrained,
  totalModels,
  elapsedTime,
  estimatedTimeRemaining,
}: PipelineExecutionProps) {
  const [steps, setSteps] = useState<PipelineStep[]>(pipelineSteps)

  useEffect(() => {
    if (isRunning) {
      // Simulate step progression
      const interval = setInterval(() => {
        setSteps(prevSteps => {
          const currentIndex = prevSteps.findIndex(step => step.status === 'running')
          if (currentIndex >= 0) {
            const newSteps = [...prevSteps]
            newSteps[currentIndex] = { ...newSteps[currentIndex], status: 'completed' }
            
            // Start next step
            if (currentIndex + 1 < newSteps.length) {
              newSteps[currentIndex + 1] = { ...newSteps[currentIndex + 1], status: 'running' }
            }
            
            return newSteps
          }
          return prevSteps
        })
      }, 2000)

      return () => clearInterval(interval)
    }
  }, [isRunning])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-6">
      {/* Execution Control */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <PlayIcon className="h-5 w-5" />
            <span>Pipeline Execution</span>
          </CardTitle>
          <CardDescription>
            Start and monitor your autonomous ML pipeline
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              {!isRunning ? (
                <Button
                  onClick={onStartPipeline}
                  size="lg"
                  className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
                >
                  <PlayIcon className="h-5 w-5 mr-2" />
                  Start Pipeline
                </Button>
              ) : (
                <Button
                  onClick={onStopPipeline}
                  variant="destructive"
                  size="lg"
                >
                  <StopIcon className="h-5 w-5 mr-2" />
                  Stop Pipeline
                </Button>
              )}
            </div>

            <div className="text-right">
              <div className="text-sm text-muted-foreground">Status</div>
              <Badge variant={isRunning ? "default" : "secondary"}>
                {isRunning ? "Running" : "Ready"}
              </Badge>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Overall Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-3" />
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="text-lg font-semibold">{modelsTrained}/{totalModels}</div>
              <div className="text-xs text-muted-foreground">Models Trained</div>
            </div>
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="text-lg font-semibold">{formatTime(elapsedTime)}</div>
              <div className="text-xs text-muted-foreground">Elapsed Time</div>
            </div>
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="text-lg font-semibold">{formatTime(estimatedTimeRemaining)}</div>
              <div className="text-xs text-muted-foreground">Est. Remaining</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Pipeline Steps */}
      <Card>
        <CardHeader>
          <CardTitle>Pipeline Steps</CardTitle>
          <CardDescription>
            Real-time progress of each pipeline stage
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-4">
            {steps.map((step, index) => {
              const Icon = step.icon
              
              return (
                <motion.div
                  key={step.id}
                  className={cn(
                    "flex items-center space-x-4 p-4 rounded-lg border transition-all",
                    step.status === 'running' && "border-primary bg-primary/5",
                    step.status === 'completed' && "border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950/20",
                    step.status === 'error' && "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950/20",
                    step.status === 'pending' && "border-input"
                  )}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div className="flex-shrink-0">
                    {step.status === 'completed' ? (
                      <CheckCircleIcon className="h-6 w-6 text-green-600" />
                    ) : step.status === 'error' ? (
                      <ExclamationTriangleIcon className="h-6 w-6 text-red-600" />
                    ) : step.status === 'running' ? (
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      >
                        <Icon className="h-6 w-6 text-primary" />
                      </motion.div>
                    ) : (
                      <Icon className="h-6 w-6 text-muted-foreground" />
                    )}
                  </div>

                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <h3 className="font-medium">{step.name}</h3>
                      <Badge
                        variant={
                          step.status === 'completed' ? 'success' :
                          step.status === 'running' ? 'default' :
                          step.status === 'error' ? 'destructive' : 'secondary'
                        }
                      >
                        {step.status}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">{step.description}</p>
                  </div>

                  {step.status === 'running' && (
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                    </div>
                  )}
                </motion.div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
