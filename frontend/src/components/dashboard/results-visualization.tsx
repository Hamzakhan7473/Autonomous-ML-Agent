"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { 
  TrophyIcon,
  ChartBarIcon,
  EyeIcon,
  ArrowDownTrayIcon,
  ArrowTrendingUpIcon,
  ClockIcon,
  CpuChipIcon
} from "@heroicons/react/24/outline"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

interface ModelResult {
  name: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  trainingTime: number
  crossValScore: number
  crossValStd: number
}

interface ResultsVisualizationProps {
  results: {
    bestModel: string
    leaderboard: ModelResult[]
    featureImportance: Record<string, number>
    modelInsights: string
    trainingTime: number
    totalIterations: number
  } | null
}

export function ResultsVisualization({ results }: ResultsVisualizationProps) {
  const [selectedMetric, setSelectedMetric] = useState<'accuracy' | 'precision' | 'recall' | 'f1'>('accuracy')
  const [showFeatureImportance, setShowFeatureImportance] = useState(true)

  if (!results) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center">
            <ChartBarIcon className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">No results available yet</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const bestModel = results.leaderboard[0]
  const sortedModels = [...results.leaderboard].sort((a, b) => b[selectedMetric] - a[selectedMetric])

  return (
    <div className="space-y-6">
      {/* Results Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrophyIcon className="h-5 w-5" />
            <span>Results Overview</span>
          </CardTitle>
          <CardDescription>
            Summary of your autonomous ML pipeline results
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <motion.div
              className="text-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950/20 dark:to-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.1 }}
            >
              <CpuChipIcon className="h-8 w-8 mx-auto mb-2 text-blue-600" />
              <div className="text-lg font-semibold text-blue-900 dark:text-blue-100">
                {results.bestModel}
              </div>
              <div className="text-xs text-blue-700 dark:text-blue-300">Best Model</div>
            </motion.div>

            <motion.div
              className="text-center p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950/20 dark:to-green-900/20 rounded-lg border border-green-200 dark:border-green-800"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2 }}
            >
              <ArrowTrendingUpIcon className="h-8 w-8 mx-auto mb-2 text-green-600" />
              <div className="text-lg font-semibold text-green-900 dark:text-green-100">
                {(bestModel.accuracy * 100).toFixed(2)}%
              </div>
              <div className="text-xs text-green-700 dark:text-green-300">Best Accuracy</div>
            </motion.div>

            <motion.div
              className="text-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950/20 dark:to-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.3 }}
            >
              <ClockIcon className="h-8 w-8 mx-auto mb-2 text-purple-600" />
              <div className="text-lg font-semibold text-purple-900 dark:text-purple-100">
                {Math.round(results.trainingTime)}s
              </div>
              <div className="text-xs text-purple-700 dark:text-purple-300">Training Time</div>
            </motion.div>

            <motion.div
              className="text-center p-4 bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-950/20 dark:to-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.4 }}
            >
              <ChartBarIcon className="h-8 w-8 mx-auto mb-2 text-orange-600" />
              <div className="text-lg font-semibold text-orange-900 dark:text-orange-100">
                {results.totalIterations}
              </div>
              <div className="text-xs text-orange-700 dark:text-orange-300">Models Trained</div>
            </motion.div>
          </div>
        </CardContent>
      </Card>

      {/* Model Leaderboard */}
      <Card>
        <CardHeader>
          <CardTitle>Model Leaderboard</CardTitle>
          <CardDescription>
            Performance comparison of all trained models
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          {/* Metric Selector */}
          <div className="flex space-x-2 mb-6">
            {(['accuracy', 'precision', 'recall', 'f1'] as const).map((metric) => (
              <Button
                key={metric}
                variant={selectedMetric === metric ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedMetric(metric)}
              >
                {metric.charAt(0).toUpperCase() + metric.slice(1)}
              </Button>
            ))}
          </div>

          {/* Model Rankings */}
          <div className="space-y-3">
            {sortedModels.map((model, index) => (
              <motion.div
                key={model.name}
                className={cn(
                  "flex items-center justify-between p-4 rounded-lg border transition-all",
                  index === 0 ? "border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-950/20" :
                  index < 3 ? "border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950/20" :
                  "border-input"
                )}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="flex items-center space-x-4">
                  <div className={cn(
                    "flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold",
                    index === 0 ? "bg-yellow-500 text-white" :
                    index < 3 ? "bg-green-500 text-white" :
                    "bg-muted text-muted-foreground"
                  )}>
                    {index + 1}
                  </div>
                  
                  <div>
                    <h3 className="font-medium">{model.name}</h3>
                    <p className="text-sm text-muted-foreground">
                      Training time: {Math.round(model.trainingTime)}s
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="text-lg font-semibold">
                      {(model[selectedMetric] * 100).toFixed(2)}%
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)}
                    </div>
                  </div>
                  
                  {index < 3 && (
                    <Badge variant={index === 0 ? "default" : "success"}>
                      {index === 0 ? "🥇 Best" : index === 1 ? "🥈 2nd" : "🥉 3rd"}
                    </Badge>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Feature Importance */}
      {showFeatureImportance && results.featureImportance && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <EyeIcon className="h-5 w-5" />
                <span>Feature Importance</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowFeatureImportance(false)}
              >
                Hide
              </Button>
            </CardTitle>
            <CardDescription>
              Most influential features in the best model
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            <div className="space-y-3">
              {Object.entries(results.featureImportance)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 10)
                .map(([feature, importance], index) => (
                  <motion.div
                    key={feature}
                    className="space-y-2"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">{feature}</span>
                      <span className="text-muted-foreground">
                        {(importance * 100).toFixed(2)}%
                      </span>
                    </div>
                    <Progress value={importance * 100} className="h-2" />
                  </motion.div>
                ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* AI Insights */}
      <Card>
        <CardHeader>
          <CardTitle>AI-Generated Insights</CardTitle>
          <CardDescription>
            Intelligent analysis and recommendations from the AI agent
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <p className="text-sm leading-relaxed">
              {results.modelInsights}
            </p>
          </div>
          
          <div className="flex justify-end mt-4">
            <Button variant="outline" size="sm">
              <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
              Export Insights
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
