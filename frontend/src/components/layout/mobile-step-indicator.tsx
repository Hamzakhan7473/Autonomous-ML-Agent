"use client"

import { motion } from "framer-motion"
import { 
  CloudArrowUpIcon,
  Cog6ToothIcon,
  PlayIcon,
  ChartBarIcon,
  DocumentTextIcon,
  CheckCircleIcon
} from "@heroicons/react/24/outline"
import { cn } from "@/lib/utils"

interface MobileStepIndicatorProps {
  currentStep: number
  onStepClick: (step: number) => void
}

const steps = [
  { id: 0, name: "Upload", icon: CloudArrowUpIcon },
  { id: 1, name: "Configure", icon: Cog6ToothIcon },
  { id: 2, name: "Run", icon: PlayIcon },
  { id: 3, name: "Results", icon: ChartBarIcon },
  { id: 4, name: "Insights", icon: DocumentTextIcon },
]

export function MobileStepIndicator({ currentStep, onStepClick }: MobileStepIndicatorProps) {
  return (
    <div className="lg:hidden bg-background border-b">
      <div className="px-4 py-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-muted-foreground">Pipeline Progress</h3>
          <span className="text-xs text-muted-foreground">
            {currentStep + 1} of {steps.length}
          </span>
        </div>
        
        <div className="mt-3 flex items-center justify-between">
          {steps.map((step, index) => {
            const isActive = currentStep === step.id
            const isCompleted = currentStep > step.id
            const Icon = step.icon

            return (
              <motion.button
                key={step.id}
                onClick={() => onStepClick(step.id)}
                className={cn(
                  "flex flex-col items-center space-y-1 p-2 rounded-lg transition-all",
                  isActive ? "bg-primary text-primary-foreground" : "hover:bg-accent"
                )}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <div className="relative">
                  {isCompleted ? (
                    <CheckCircleIcon className="h-5 w-5 text-green-600" />
                  ) : (
                    <Icon className={cn(
                      "h-5 w-5",
                      isActive ? "text-primary-foreground" : "text-muted-foreground"
                    )} />
                  )}
                  
                  {isActive && (
                    <motion.div
                      className="absolute -top-1 -right-1 h-2 w-2 bg-primary-foreground rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                    />
                  )}
                </div>
                
                <span className={cn(
                  "text-xs font-medium",
                  isActive ? "text-primary-foreground" : "text-muted-foreground"
                )}>
                  {step.name}
                </span>
              </motion.button>
            )
          })}
        </div>
        
        {/* Progress bar */}
        <div className="mt-3">
          <div className="w-full bg-secondary rounded-full h-1">
            <motion.div
              className="bg-primary h-1 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
