"use client"

import { useState, useCallback } from "react"
import { motion } from "framer-motion"
import { 
  CloudArrowUpIcon,
  DocumentIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XMarkIcon
} from "@heroicons/react/24/outline"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface DataUploadProps {
  onFileUpload: (file: File) => void
  uploadedFile: File | null
  onRemoveFile: () => void
}

export function DataUpload({ onFileUpload, uploadedFile, onRemoveFile }: DataUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    
    const files = Array.from(e.dataTransfer.files)
    const csvFile = files.find(file => file.type === 'text/csv' || file.name.endsWith('.csv'))
    
    if (csvFile) {
      handleFileUpload(csvFile)
    }
  }, [])

  const handleFileUpload = async (file: File) => {
    setIsUploading(true)
    try {
      // Simulate upload delay
      await new Promise(resolve => setTimeout(resolve, 1000))
      onFileUpload(file)
    } finally {
      setIsUploading(false)
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileUpload(file)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <CloudArrowUpIcon className="h-5 w-5" />
          <span>Upload Dataset</span>
        </CardTitle>
        <CardDescription>
          Upload your CSV dataset to begin the autonomous ML pipeline
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        {!uploadedFile ? (
          <motion.div
            className={cn(
              "relative border-2 border-dashed rounded-lg p-8 text-center transition-colors",
              isDragOver 
                ? "border-primary bg-primary/5" 
                : "border-muted-foreground/25 hover:border-primary/50"
            )}
            onDrop={handleDrop}
            onDragOver={(e) => {
              e.preventDefault()
              setIsDragOver(true)
            }}
            onDragLeave={() => setIsDragOver(false)}
            whileHover={{ scale: 1.01 }}
            transition={{ duration: 0.2 }}
          >
            <input
              type="file"
              accept=".csv"
              onChange={handleFileInput}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              disabled={isUploading}
            />
            
            <motion.div
              animate={{ 
                scale: isDragOver ? 1.1 : 1,
                rotate: isDragOver ? 5 : 0 
              }}
              transition={{ duration: 0.2 }}
            >
              <CloudArrowUpIcon className="h-12 w-12 mx-auto mb-4 text-slate-600 dark:text-slate-400" />
            </motion.div>
            
            <h3 className="text-lg font-semibold mb-2">
              {isUploading ? "Uploading..." : "Drop your CSV file here"}
            </h3>
            
            <p className="text-slate-600 dark:text-slate-400 mb-4">
              or click to browse files
            </p>
            
            <div className="flex items-center justify-center space-x-4 text-sm text-slate-600 dark:text-slate-400">
              <div className="flex items-center space-x-1">
                <DocumentIcon className="h-4 w-4" />
                <span>CSV format</span>
              </div>
              <div className="flex items-center space-x-1">
                <ExclamationTriangleIcon className="h-4 w-4" />
                <span>Max 100MB</span>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <div className="flex items-center justify-between p-4 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
              <div className="flex items-center space-x-3">
                <CheckCircleIcon className="h-5 w-5 text-green-600" />
                <div>
                  <p className="font-medium text-green-900 dark:text-green-100">
                    {uploadedFile.name}
                  </p>
                  <p className="text-sm text-green-700 dark:text-green-300">
                    {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              
              <Button
                variant="ghost"
                size="icon"
                onClick={onRemoveFile}
                className="text-green-600 hover:text-green-700 hover:bg-green-100 dark:hover:bg-green-900/20"
              >
                <XMarkIcon className="h-4 w-4" />
              </Button>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="text-center p-3 bg-slate-100 dark:bg-slate-800 rounded-lg">
                <p className="text-sm text-slate-600 dark:text-slate-400">File Type</p>
                <Badge variant="secondary">CSV</Badge>
              </div>
              <div className="text-center p-3 bg-slate-100 dark:bg-slate-800 rounded-lg">
                <p className="text-sm text-slate-600 dark:text-slate-400">Status</p>
                <Badge variant="success">Ready</Badge>
              </div>
            </div>
          </motion.div>
        )}
      </CardContent>
    </Card>
  )
}
