# Frontend Integration Guide

This guide explains how to set up and integrate the frontend with the Autonomous ML Agent backend.

## 🎯 Overview

The frontend is a modern, responsive web application built with Next.js that provides a complete user interface for the Autonomous ML Agent. It includes:

- **Dataset Upload & Analysis**: Drag-and-drop interface for data upload
- **Pipeline Configuration**: Interactive configuration with presets
- **Real-time Monitoring**: Live progress tracking during training
- **Results Visualization**: Comprehensive charts and analysis
- **Model Deployment**: Production deployment capabilities

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# From the project root
./setup_frontend.sh
```

### 2. Start the Backend

```bash
# Start the FastAPI backend
cd /Users/hamzakhan/autonomous_ml_agent
source venv/bin/activate
python -m src.service.app
```

### 3. Start the Frontend

```bash
# In a new terminal
cd frontend
npm run dev
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🏗️ Architecture

### Frontend Stack
- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Charts**: Recharts
- **Icons**: Heroicons

### Backend Integration
- **API Client**: Custom fetch-based client with TypeScript
- **Real-time Updates**: Polling-based status updates
- **Error Handling**: Comprehensive error management
- **File Upload**: Drag-and-drop with progress tracking

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── dashboard/         # Main dashboard page
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx          # Landing page
│   ├── components/            # React components
│   │   ├── dashboard/        # Dashboard components
│   │   └── ui/               # Reusable UI components
│   └── lib/                  # Utilities
│       ├── api.ts           # API client
│       ├── hooks.ts         # Custom React hooks
│       └── types.ts         # TypeScript types
├── package.json
├── tailwind.config.ts
└── README.md
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file in the frontend directory:

```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Development Configuration
NODE_ENV=development
```

### API Configuration

The API client is configured in `src/lib/api.ts`:

```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
```

## 🔌 API Integration

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze dataset |
| `/pipeline/run` | POST | Start pipeline |
| `/pipeline/status/{task_id}` | GET | Get task status |
| `/predict` | POST | Make predictions |
| `/model/info/{task_id}` | GET | Get model info |
| `/tasks` | GET | List all tasks |

### Usage Examples

#### Upload and Analyze Dataset

```typescript
import { apiClient } from '@/lib/api';

const file = new File([data], 'dataset.csv', { type: 'text/csv' });
const datasetInfo = await apiClient.analyzeDataset(file);
console.log('Dataset shape:', datasetInfo.shape);
```

#### Start Pipeline Execution

```typescript
const response = await apiClient.runPipeline({
  dataset_path: '/path/to/dataset.csv',
  target_column: 'target',
  time_budget: 3600,
  optimization_metric: 'auto'
});
console.log('Task ID:', response.task_id);
```

#### Monitor Pipeline Progress

```typescript
const cleanup = await apiClient.subscribeToTaskStatus(
  taskId,
  (status) => {
    console.log('Progress:', status.progress * 100 + '%');
  },
  (error) => {
    console.error('Error:', error);
  }
);
```

## 🎨 Components

### Dashboard Components

#### DatasetUpload
- Drag-and-drop file upload
- File validation and analysis
- Progress tracking
- Error handling

#### PipelineConfig
- Interactive configuration panel
- Preset configurations
- Custom parameter settings
- Save/load configurations

#### PipelineExecution
- Real-time progress monitoring
- Status updates
- Error handling
- Stop/reset functionality

#### ResultsVisualization
- Model leaderboard
- Performance charts
- Feature importance
- Prediction interface

### UI Components

#### LoadingSpinner
- Animated loading indicator
- Multiple sizes
- Customizable text

#### Alert
- Success, warning, error, info types
- Dismissible
- Animated transitions

## 🔄 Real-time Updates

The frontend implements real-time updates using polling:

```typescript
// Poll every 2 seconds for status updates
const cleanup = await apiClient.subscribeToTaskStatus(
  taskId,
  onUpdate,
  onError,
  2000
);
```

## 📊 Data Visualization

### Charts Used

1. **Bar Charts**: Model performance comparison
2. **Line Charts**: Metric trends over time
3. **Pie Charts**: Model distribution
4. **Scatter Charts**: Training time vs performance
5. **Horizontal Bar Charts**: Feature importance

### Chart Library

Using Recharts for responsive, interactive charts:

```typescript
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
```

## 🚀 Deployment

### Development

```bash
npm run dev
```

### Production Build

```bash
npm run build
npm start
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Vercel

1. Connect GitHub repository
2. Set environment variables
3. Deploy automatically

## 🧪 Testing

### Run Tests

```bash
npm test
```

### Type Checking

```bash
npm run type-check
```

### Linting

```bash
npm run lint
npm run lint:fix
```

## 🔧 Customization

### Adding New Features

1. Create component in appropriate directory
2. Add TypeScript types in `lib/types.ts`
3. Update API client if needed
4. Add to dashboard navigation

### Styling

- Use Tailwind CSS classes
- Follow design system in `lib/config.ts`
- Maintain responsive design
- Ensure accessibility

### API Extensions

1. Add new endpoints to API client
2. Update TypeScript types
3. Create corresponding UI components
4. Update documentation

## 🐛 Troubleshooting

### Common Issues

#### CORS Errors
- Ensure backend has CORS middleware enabled
- Check API URL configuration

#### Build Failures
- Run `npm run type-check` to identify TypeScript errors
- Check for missing dependencies

#### API Connection Issues
- Verify backend is running on correct port
- Check environment variables
- Test API endpoints directly

### Debug Mode

Enable debug logging:

```typescript
// In api.ts
const DEBUG = process.env.NODE_ENV === 'development';
```

## 📚 Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Framer Motion](https://www.framer.com/motion/)
- [Recharts](https://recharts.org/)
- [Heroicons](https://heroicons.com/)

## 🤝 Contributing

1. Follow the existing code style
2. Add TypeScript types for new features
3. Update documentation
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is licensed under the MIT License.
