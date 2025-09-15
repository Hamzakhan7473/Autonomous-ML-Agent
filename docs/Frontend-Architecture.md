# ⚛️ Frontend Architecture

## **Overview**

The Autonomous ML Agent frontend is built with **Next.js 15** using the App Router, TypeScript, and Tailwind CSS. It provides a modern, responsive web interface for interacting with the ML pipeline.

## **Technology Stack**

- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe JavaScript development
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Animation and gesture library
- **React Hooks**: State management and side effects

## **Project Structure**

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── globals.css        # Global styles
│   │   ├── layout.tsx         # Root layout component
│   │   ├── page.tsx           # Landing page
│   │   └── dashboard/         # Dashboard pages
│   │       └── page.tsx       # Dashboard page
│   ├── components/            # Reusable components
│   │   ├── dashboard/         # Dashboard-specific components
│   │   ├── landing/           # Landing page components
│   │   ├── layout/            # Layout components
│   │   └── ui/               # Base UI components
│   └── lib/                  # Utilities and configuration
│       ├── api.ts            # API client functions
│       ├── config.ts         # App configuration
│       └── types.ts          # TypeScript type definitions
├── public/                   # Static assets
├── package.json             # Dependencies and scripts
├── tailwind.config.ts       # Tailwind configuration
├── tsconfig.json           # TypeScript configuration
└── next.config.ts          # Next.js configuration
```

## **Component Architecture**

### **Layout Components**

#### **Root Layout (`app/layout.tsx`)**
```typescript
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Header />
        <main className="min-h-screen bg-gray-50">
          {children}
        </main>
      </body>
    </html>
  )
}
```

#### **Header Component (`components/layout/header.tsx`)**
```typescript
export default function Header() {
  return (
    <header className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Logo />
          <Navigation />
        </div>
      </div>
    </header>
  )
}
```

#### **Navigation Component (`components/layout/navigation.tsx`)**
```typescript
export default function Navigation() {
  const [activeTab, setActiveTab] = useState('dashboard')
  
  return (
    <nav className="flex space-x-8">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          className={cn(
            "px-3 py-2 rounded-md text-sm font-medium transition-colors",
            activeTab === tab.id
              ? "bg-blue-100 text-blue-700"
              : "text-gray-500 hover:text-gray-700"
          )}
        >
          {tab.name}
        </button>
      ))}
    </nav>
  )
}
```

### **Dashboard Components**

#### **Main Dashboard (`components/dashboard/dashboard.tsx`)**
```typescript
export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('upload')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadedFilePath, setUploadedFilePath] = useState<string | null>(null)

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        <div className="lg:col-span-1">
          <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
        </div>
        
        <div className="lg:col-span-3">
          <div className="bg-white rounded-lg shadow">
            {activeTab === 'upload' && (
              <DataUpload onFileSelect={setSelectedFile} />
            )}
            {activeTab === 'config' && (
              <PipelineConfig 
                selectedFile={selectedFile}
                onUpload={setUploadedFilePath}
              />
            )}
            {activeTab === 'execution' && (
              <PipelineExecution 
                uploadedFilePath={uploadedFilePath}
                selectedFile={selectedFile}
              />
            )}
            {activeTab === 'results' && (
              <ResultsVisualization />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
```

#### **Data Upload Component (`components/dashboard/data-upload.tsx`)**
```typescript
export default function DataUpload({ onFileSelect }: { onFileSelect: (file: File) => void }) {
  const [dragActive, setDragActive] = useState(false)

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    
    const files = e.dataTransfer.files
    if (files && files[0]) {
      onFileSelect(files[0])
    }
  }

  return (
    <div
      className={cn(
        "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
        dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300"
      )}
      onDragEnter={() => setDragActive(true)}
      onDragLeave={() => setDragActive(false)}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      <UploadIcon className="mx-auto h-12 w-12 text-gray-400" />
      <p className="mt-2 text-sm text-gray-600">
        Drag and drop your CSV file here, or click to browse
      </p>
    </div>
  )
}
```

#### **Pipeline Config Component (`components/dashboard/pipeline-config.tsx`)**
```typescript
export default function PipelineConfig({ 
  selectedFile, 
  onUpload 
}: { 
  selectedFile: File | null
  onUpload: (filePath: string) => void 
}) {
  const [config, setConfig] = useState({
    timeBudget: 30,
    optimizationMetric: 'accuracy',
    randomState: 42
  })

  const uploadFile = async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${apiConfig.api.baseUrl}/analyze`, {
      method: 'POST',
      body: formData,
    })

    if (response.ok) {
      const data = await response.json()
      onUpload(data.dataset_path)
    }
  }

  return (
    <div className="p-6">
      <h2 className="text-lg font-medium text-gray-900 mb-4">
        Pipeline Configuration
      </h2>
      
      {selectedFile && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-medium text-gray-900">Selected File</h3>
          <p className="text-sm text-gray-600">{selectedFile.name}</p>
          <button
            onClick={() => uploadFile(selectedFile)}
            className="mt-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Upload and Configure
          </button>
        </div>
      )}

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Time Budget (minutes)
          </label>
          <input
            type="number"
            value={config.timeBudget}
            onChange={(e) => setConfig({...config, timeBudget: parseInt(e.target.value)})}
            className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Optimization Metric
          </label>
          <select
            value={config.optimizationMetric}
            onChange={(e) => setConfig({...config, optimizationMetric: e.target.value})}
            className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="accuracy">Accuracy</option>
            <option value="f1">F1 Score</option>
            <option value="precision">Precision</option>
            <option value="recall">Recall</option>
            <option value="roc_auc">ROC AUC</option>
          </select>
        </div>
      </div>
    </div>
  )
}
```

#### **Pipeline Execution Component (`components/dashboard/pipeline-execution.tsx`)**
```typescript
export default function PipelineExecution({ 
  uploadedFilePath, 
  selectedFile 
}: { 
  uploadedFilePath: string | null
  selectedFile: File | null 
}) {
  const [taskId, setTaskId] = useState<string | null>(null)
  const [status, setStatus] = useState<TaskStatus>('idle')
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('')

  const startExecution = async () => {
    if (!uploadedFilePath) return

    const response = await fetch(`${apiConfig.api.baseUrl}/pipeline/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_path: uploadedFilePath,
        target_column: '',
        time_budget: 1800,
        optimization_metric: 'accuracy'
      })
    })

    if (response.ok) {
      const data = await response.json()
      setTaskId(data.task_id)
      setStatus('running')
      startStatusPolling(data.task_id)
    }
  }

  const startStatusPolling = (taskId: string) => {
    const interval = setInterval(async () => {
      const response = await fetch(`${apiConfig.api.baseUrl}/pipeline/status/${taskId}`)
      if (response.ok) {
        const data = await response.json()
        setStatus(data.status)
        setProgress(data.progress)
        setMessage(data.message)

        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval)
        }
      }
    }, 2000)
  }

  return (
    <div className="p-6">
      <h2 className="text-lg font-medium text-gray-900 mb-4">
        Pipeline Execution
      </h2>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Status:</span>
          <Badge variant={getStatusVariant(status)}>
            {status}
          </Badge>
        </div>

        {status === 'running' && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Progress:</span>
              <span className="text-sm text-gray-500">{Math.round(progress * 100)}%</span>
            </div>
            <Progress value={progress * 100} className="w-full" />
          </div>
        )}

        <div>
          <span className="text-sm font-medium text-gray-700">Message:</span>
          <p className="text-sm text-gray-600 mt-1">{message}</p>
        </div>

        <button
          onClick={startExecution}
          disabled={!uploadedFilePath || status === 'running'}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {status === 'running' ? 'Running...' : 'Start Pipeline'}
        </button>
      </div>
    </div>
  )
}
```

#### **Results Visualization Component (`components/dashboard/results-visualization.tsx`)**
```typescript
export default function ResultsVisualization() {
  const [results, setResults] = useState<ResultsData | null>(null)

  return (
    <div className="p-6">
      <h2 className="text-lg font-medium text-gray-900 mb-4">
        Results Visualization
      </h2>

      {results ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Best Model</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{results.bestModel}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Best Score</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{results.bestScore.toFixed(4)}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Execution Time</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{results.executionTime.toFixed(1)}s</p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Model Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Model
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Score
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Time
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {results.modelResults.map((result, index) => (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {result.modelName}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {result.score.toFixed(4)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {result.time.toFixed(2)}s
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-gray-500">No results available yet</p>
        </div>
      )}
    </div>
  )
}
```

### **UI Components**

#### **Base UI Components (`components/ui/`)**

The UI components follow a consistent design system:

```typescript
// Button Component
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
}

export function Button({ variant = 'primary', size = 'md', ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-md font-medium transition-colors",
        variants[variant],
        sizes[size]
      )}
      {...props}
    />
  )
}

// Card Component
export function Card({ children, className, ...props }: CardProps) {
  return (
    <div className={cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)} {...props}>
      {children}
    </div>
  )
}

// Badge Component
export function Badge({ variant = 'default', children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        badgeVariants[variant]
      )}
      {...props}
    >
      {children}
    </span>
  )
}
```

## **State Management**

### **Local State with React Hooks**

The application uses React hooks for state management:

```typescript
// Dashboard state
const [activeTab, setActiveTab] = useState('upload')
const [selectedFile, setSelectedFile] = useState<File | null>(null)
const [uploadedFilePath, setUploadedFilePath] = useState<string | null>(null)

// Pipeline execution state
const [taskId, setTaskId] = useState<string | null>(null)
const [status, setStatus] = useState<TaskStatus>('idle')
const [progress, setProgress] = useState(0)
```

### **Custom Hooks**

```typescript
// usePipeline hook for pipeline management
export function usePipeline() {
  const [taskId, setTaskId] = useState<string | null>(null)
  const [status, setStatus] = useState<TaskStatus>('idle')
  const [progress, setProgress] = useState(0)

  const startPipeline = async (config: PipelineConfig) => {
    // Implementation
  }

  const pollStatus = async (taskId: string) => {
    // Implementation
  }

  return {
    taskId,
    status,
    progress,
    startPipeline,
    pollStatus
  }
}

// useFileUpload hook for file management
export function useFileUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadedFilePath, setUploadedFilePath] = useState<string | null>(null)

  const uploadFile = async (file: File) => {
    // Implementation
  }

  return {
    selectedFile,
    uploadedFilePath,
    setSelectedFile,
    uploadFile
  }
}
```

## **API Integration**

### **API Client (`lib/api.ts`)**

```typescript
const apiConfig = {
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  }
}

export async function uploadDataset(file: File): Promise<DatasetInfo> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${apiConfig.api.baseUrl}/analyze`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error('Upload failed')
  }

  return response.json()
}

export async function runPipeline(config: PipelineConfig): Promise<PipelineResponse> {
  const response = await fetch(`${apiConfig.api.baseUrl}/pipeline/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })

  if (!response.ok) {
    throw new Error('Pipeline execution failed')
  }

  return response.json()
}

export async function getTaskStatus(taskId: string): Promise<TaskStatus> {
  const response = await fetch(`${apiConfig.api.baseUrl}/pipeline/status/${taskId}`)

  if (!response.ok) {
    if (response.status === 404) {
      return { task_id: taskId, status: 'not_found', progress: 0, message: 'Task not found' }
    }
    throw new Error('Failed to get task status')
  }

  return response.json()
}
```

## **Styling and Design System**

### **Tailwind Configuration (`tailwind.config.ts`)**

```typescript
import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          500: '#6b7280',
          900: '#111827',
        }
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      }
    },
  },
  plugins: [],
}

export default config
```

### **Design Tokens**

```typescript
// Design system constants
export const colors = {
  primary: {
    50: '#eff6ff',
    500: '#3b82f6',
    600: '#2563eb',
    700: '#1d4ed8',
  },
  gray: {
    50: '#f9fafb',
    100: '#f3f4f6',
    500: '#6b7280',
    900: '#111827',
  }
}

export const spacing = {
  xs: '0.25rem',
  sm: '0.5rem',
  md: '1rem',
  lg: '1.5rem',
  xl: '2rem',
}

export const typography = {
  fontFamily: {
    sans: ['Inter', 'system-ui', 'sans-serif'],
  },
  fontSize: {
    xs: '0.75rem',
    sm: '0.875rem',
    base: '1rem',
    lg: '1.125rem',
    xl: '1.25rem',
  }
}
```

## **Responsive Design**

### **Mobile-First Approach**

```typescript
// Responsive grid layout
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* Content */}
</div>

// Responsive text sizing
<h1 className="text-2xl md:text-3xl lg:text-4xl font-bold">
  Autonomous ML Agent
</h1>

// Responsive spacing
<div className="p-4 md:p-6 lg:p-8">
  {/* Content */}
</div>
```

### **Breakpoint Strategy**

- **Mobile**: `< 768px` - Single column layout
- **Tablet**: `768px - 1024px` - Two column layout
- **Desktop**: `> 1024px` - Multi-column layout

## **Performance Optimization**

### **Code Splitting**

```typescript
// Dynamic imports for heavy components
const ResultsVisualization = dynamic(() => import('./results-visualization'), {
  loading: () => <LoadingSpinner />,
})

// Lazy loading for dashboard tabs
const DashboardTab = lazy(() => import(`./tabs/${tabName}`))
```

### **Image Optimization**

```typescript
import Image from 'next/image'

// Optimized image loading
<Image
  src="/logo.svg"
  alt="Autonomous ML Agent"
  width={200}
  height={50}
  priority
/>
```

### **Bundle Analysis**

```bash
# Analyze bundle size
npm run build
npm run analyze
```

## **Testing Strategy**

### **Unit Tests**

```typescript
// Component testing with React Testing Library
import { render, screen, fireEvent } from '@testing-library/react'
import { DataUpload } from './data-upload'

test('should handle file drop', () => {
  const onFileSelect = jest.fn()
  render(<DataUpload onFileSelect={onFileSelect} />)
  
  const dropZone = screen.getByText(/drag and drop/i)
  fireEvent.drop(dropZone, {
    dataTransfer: {
      files: [new File(['test'], 'test.csv', { type: 'text/csv' })]
    }
  })
  
  expect(onFileSelect).toHaveBeenCalled()
})
```

### **Integration Tests**

```typescript
// API integration testing
import { uploadDataset, runPipeline } from '../lib/api'

test('should upload dataset and run pipeline', async () => {
  const file = new File(['test,data\n1,2'], 'test.csv', { type: 'text/csv' })
  
  const datasetInfo = await uploadDataset(file)
  expect(datasetInfo.filename).toBe('test.csv')
  
  const pipelineResponse = await runPipeline({
    dataset_path: datasetInfo.dataset_path,
    target_column: datasetInfo.target_column
  })
  expect(pipelineResponse.status).toBe('running')
})
```

## **Deployment**

### **Build Process**

```bash
# Install dependencies
npm install

# Build for production
npm run build

# Start production server
npm start
```

### **Environment Variables**

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=Autonomous ML Agent
```

### **Docker Deployment**

```dockerfile
FROM node:18-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine AS builder
WORKDIR /app
COPY . .
COPY --from=deps /app/node_modules ./node_modules
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
```

## **Best Practices**

### **1. Component Design**
- Keep components small and focused
- Use TypeScript for type safety
- Follow single responsibility principle
- Implement proper error boundaries

### **2. State Management**
- Use local state when possible
- Implement custom hooks for complex logic
- Avoid prop drilling with context
- Use proper dependency arrays in useEffect

### **3. Performance**
- Implement code splitting
- Use React.memo for expensive components
- Optimize re-renders with useMemo/useCallback
- Implement proper loading states

### **4. Accessibility**
- Use semantic HTML elements
- Implement proper ARIA attributes
- Ensure keyboard navigation
- Test with screen readers

### **5. SEO and Performance**
- Use Next.js Image component
- Implement proper meta tags
- Optimize Core Web Vitals
- Use proper heading hierarchy

---

This frontend architecture provides a solid foundation for the Autonomous ML Agent's user interface, ensuring a modern, responsive, and maintainable web application.

