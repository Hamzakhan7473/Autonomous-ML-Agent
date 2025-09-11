"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { 
  SunIcon, 
  MoonIcon, 
  Cog6ToothIcon,
  SparklesIcon,
  ChartBarIcon
} from "@heroicons/react/24/outline"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface HeaderProps {
  onToggleTheme: () => void
  isDark: boolean
}

export function Header({ onToggleTheme, isDark }: HeaderProps) {
  const [isConnected, setIsConnected] = useState(true)

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-white/95 dark:bg-gray-900/95 backdrop-blur supports-[backdrop-filter]:bg-white/60 dark:supports-[backdrop-filter]:bg-gray-900/60">
      <div className="container flex h-16 items-center justify-between">
        {/* Logo and Title */}
        <div className="flex items-center space-x-3">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 10 }}
            className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-purple-600"
          >
            <SparklesIcon className="h-6 w-6 text-white" />
          </motion.div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Autonomous ML Agent
            </h1>
            <p className="text-xs text-slate-600 dark:text-slate-400">
              AI-Powered Machine Learning Pipeline
            </p>
          </div>
        </div>

        {/* Status and Controls */}
        <div className="flex items-center space-x-4">
          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm text-slate-600 dark:text-slate-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Model Registry Status */}
          <Badge variant="secondary" className="flex items-center space-x-1">
            <ChartBarIcon className="h-3 w-3" />
            <span>Registry Active</span>
          </Badge>

          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleTheme}
            className="relative overflow-hidden"
          >
            <motion.div
              key={isDark ? 'dark' : 'light'}
              initial={{ rotate: -180, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 180, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              {isDark ? (
                <SunIcon className="h-5 w-5" />
              ) : (
                <MoonIcon className="h-5 w-5" />
              )}
            </motion.div>
          </Button>

          {/* Settings */}
          <Button variant="ghost" size="icon">
            <Cog6ToothIcon className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </header>
  )
}
