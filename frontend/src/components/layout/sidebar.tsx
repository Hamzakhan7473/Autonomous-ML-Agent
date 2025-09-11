"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { 
  HomeIcon,
  CloudArrowUpIcon,
  Cog6ToothIcon,
  ChartBarIcon,
  DocumentTextIcon,
  PlayIcon,
  ClockIcon,
  CheckCircleIcon
} from "@heroicons/react/24/outline"
import { cn } from "@/lib/utils"

interface SidebarProps {
  currentStep: number
  onStepClick: (step: number) => void
}

const steps = [
  { id: 0, name: "Upload Data", icon: CloudArrowUpIcon, description: "Upload your dataset" },
  { id: 1, name: "Configure", icon: Cog6ToothIcon, description: "Set pipeline parameters" },
  { id: 2, name: "Run Pipeline", icon: PlayIcon, description: "Execute ML pipeline" },
  { id: 3, name: "Results", icon: ChartBarIcon, description: "View model performance" },
  { id: 4, name: "Insights", icon: DocumentTextIcon, description: "AI-generated insights" },
]

export function Sidebar({ currentStep, onStepClick }: SidebarProps) {
  return (
    <aside className="hidden lg:block w-64 border-r bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm">
      <div className="p-6">
        <h2 className="text-lg font-semibold mb-6">Pipeline Steps</h2>
        
        <nav className="space-y-2">
          {steps.map((step, index) => {
            const isActive = currentStep === step.id
            const isCompleted = currentStep > step.id
            const Icon = step.icon

            return (
              <motion.button
                key={step.id}
                onClick={() => onStepClick(step.id)}
                className={cn(
                  "w-full flex items-center space-x-3 px-3 py-3 rounded-lg text-left transition-all duration-200",
                  isActive 
                    ? "bg-primary text-white shadow-md" 
                    : "hover:bg-accent hover:text-accent-foreground",
                  isCompleted && !isActive && "bg-green-50 dark:bg-green-950/20"
                )}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex-shrink-0">
                  {isCompleted && !isActive ? (
                    <CheckCircleIcon className="h-5 w-5 text-green-600" />
                  ) : (
                    <Icon className={cn(
                      "h-5 w-5",
                      isActive ? "text-white" : "text-slate-600 dark:text-slate-400"
                    )} />
                  )}
                </div>
                
                <div className="flex-1 min-w-0">
                  <p className={cn(
                    "text-sm font-medium",
                    isActive ? "text-white" : "text-gray-700 dark:text-gray-300"
                  )}>
                    {step.name}
                  </p>
                  <p className={cn(
                    "text-xs",
                    isActive ? "text-white/80" : "text-slate-600 dark:text-slate-400"
                  )}>
                    {step.description}
                  </p>
                </div>

                {/* Progress indicator */}
                {isActive && (
                  <motion.div
                    className="h-2 w-2 rounded-full bg-primary-foreground"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                  />
                )}
              </motion.button>
            )
          })}
        </nav>

        {/* Pipeline Status */}
        <div className="mt-8 p-4 rounded-lg bg-slate-100 dark:bg-slate-800">
          <div className="flex items-center space-x-2 mb-2">
            <ClockIcon className="h-4 w-4 text-slate-600 dark:text-slate-400" />
            <span className="text-sm font-medium">Pipeline Status</span>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span>Progress</span>
              <span>{Math.round((currentStep / (steps.length - 1)) * 100)}%</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2">
              <motion.div
                className="bg-primary h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}
