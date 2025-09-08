"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { 
  Cog6ToothIcon,
  TagIcon,
  ClockIcon,
  ChartBarIcon,
  CpuChipIcon,
  SparklesIcon
} from "@heroicons/react/24/outline"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface PipelineConfigProps {
  targetColumn: string
  onTargetColumnChange: (value: string) => void
  optimizationMetric: string
  onOptimizationMetricChange: (value: string) => void
  timeBudget: number
  onTimeBudgetChange: (value: number) => void
  maxModels: number
  onMaxModelsChange: (value: number) => void
  cvFolds: number
  onCvFoldsChange: (value: number) => void
  enableEnsemble: boolean
  onEnableEnsembleChange: (value: boolean) => void
  enableInterpretability: boolean
  onEnableInterpretabilityChange: (value: boolean) => void
  enableMetaLearning: boolean
  onEnableMetaLearningChange: (value: boolean) => void
}

const optimizationMetrics = [
  { value: 'accuracy', label: 'Accuracy', description: 'Overall correctness' },
  { value: 'precision', label: 'Precision', description: 'True positives / (True positives + False positives)' },
  { value: 'recall', label: 'Recall', description: 'True positives / (True positives + False negatives)' },
  { value: 'f1', label: 'F1 Score', description: 'Harmonic mean of precision and recall' },
  { value: 'auc', label: 'AUC', description: 'Area under the ROC curve' },
]

export function PipelineConfig({
  targetColumn,
  onTargetColumnChange,
  optimizationMetric,
  onOptimizationMetricChange,
  timeBudget,
  onTimeBudgetChange,
  maxModels,
  onMaxModelsChange,
  cvFolds,
  onCvFoldsChange,
  enableEnsemble,
  onEnableEnsembleChange,
  enableInterpretability,
  onEnableInterpretabilityChange,
  enableMetaLearning,
  onEnableMetaLearningChange,
}: PipelineConfigProps) {
  return (
    <div className="space-y-6">
      {/* Basic Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Cog6ToothIcon className="h-5 w-5" />
            <span>Basic Configuration</span>
          </CardTitle>
          <CardDescription>
            Configure the core parameters for your ML pipeline
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {/* Target Column */}
          <div className="space-y-2">
            <label className="flex items-center space-x-2 text-sm font-medium">
              <TagIcon className="h-4 w-4" />
              <span>Target Column</span>
            </label>
            <input
              type="text"
              value={targetColumn}
              onChange={(e) => onTargetColumnChange(e.target.value)}
              placeholder="Enter target column name"
              className="w-full px-3 py-2 border border-input rounded-md bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            />
          </div>

          {/* Optimization Metric */}
          <div className="space-y-2">
            <label className="flex items-center space-x-2 text-sm font-medium">
              <ChartBarIcon className="h-4 w-4" />
              <span>Optimization Metric</span>
            </label>
            <div className="grid grid-cols-1 gap-2">
              {optimizationMetrics.map((metric) => (
                <motion.button
                  key={metric.value}
                  onClick={() => onOptimizationMetricChange(metric.value)}
                  className={cn(
                    "flex items-center justify-between p-3 rounded-lg border text-left transition-all",
                    optimizationMetric === metric.value
                      ? "border-primary bg-primary/5"
                      : "border-input hover:border-primary/50"
                  )}
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                >
                  <div>
                    <p className="font-medium">{metric.label}</p>
                    <p className="text-sm text-muted-foreground">{metric.description}</p>
                  </div>
                  {optimizationMetric === metric.value && (
                    <Badge variant="default">Selected</Badge>
                  )}
                </motion.button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <CpuChipIcon className="h-5 w-5" />
            <span>Advanced Settings</span>
          </CardTitle>
          <CardDescription>
            Fine-tune the pipeline behavior and performance
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {/* Time Budget */}
          <div className="space-y-2">
            <label className="flex items-center space-x-2 text-sm font-medium">
              <ClockIcon className="h-4 w-4" />
              <span>Time Budget: {timeBudget} minutes</span>
            </label>
            <input
              type="range"
              min="5"
              max="120"
              value={timeBudget}
              onChange={(e) => onTimeBudgetChange(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>5 min</span>
              <span>120 min</span>
            </div>
          </div>

          {/* Max Models */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Maximum Models: {maxModels}</label>
            <input
              type="range"
              min="3"
              max="15"
              value={maxModels}
              onChange={(e) => onMaxModelsChange(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>3</span>
              <span>15</span>
            </div>
          </div>

          {/* Cross-Validation Folds */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Cross-Validation Folds: {cvFolds}</label>
            <input
              type="range"
              min="3"
              max="10"
              value={cvFolds}
              onChange={(e) => onCvFoldsChange(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>3</span>
              <span>10</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feature Flags */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <SparklesIcon className="h-5 w-5" />
            <span>AI Features</span>
          </CardTitle>
          <CardDescription>
            Enable advanced AI-powered features
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {[
            {
              key: 'ensemble',
              label: 'Ensemble Learning',
              description: 'Combine multiple models for better performance',
              enabled: enableEnsemble,
              onChange: onEnableEnsembleChange,
            },
            {
              key: 'interpretability',
              label: 'Model Interpretability',
              description: 'Generate explanations and feature importance',
              enabled: enableInterpretability,
              onChange: onEnableInterpretabilityChange,
            },
            {
              key: 'metaLearning',
              label: 'Meta-Learning',
              description: 'Use previous experiments to improve performance',
              enabled: enableMetaLearning,
              onChange: onEnableMetaLearningChange,
            },
          ].map((feature) => (
            <motion.div
              key={feature.key}
              className="flex items-center justify-between p-3 rounded-lg border border-input hover:border-primary/50 transition-colors"
              whileHover={{ scale: 1.01 }}
            >
              <div>
                <p className="font-medium">{feature.label}</p>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </div>
              <Button
                variant={feature.enabled ? "default" : "outline"}
                size="sm"
                onClick={() => feature.onChange(!feature.enabled)}
              >
                {feature.enabled ? "Enabled" : "Disabled"}
              </Button>
            </motion.div>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}
