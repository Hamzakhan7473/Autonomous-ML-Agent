# Autonomous ML Agent Frontend

A modern, responsive frontend for the Autonomous ML Agent built with Next.js, TypeScript, and Tailwind CSS.

## Features

### 🚀 Core Features
- **Dataset Upload & Analysis**: Drag-and-drop interface for uploading CSV, Excel, and JSON files
- **Pipeline Configuration**: Interactive configuration panel with presets and custom settings
- **Real-time Execution Monitoring**: Live progress tracking with WebSocket-like updates
- **Results Visualization**: Comprehensive charts and tables for model performance analysis
- **Model Deployment**: One-click deployment to production environments

### 📊 Visualization Components
- **Leaderboard**: Ranked model performance comparison
- **Performance Charts**: Interactive charts using Recharts
- **Feature Importance**: Visual representation of feature contributions
- **Training Progress**: Real-time training progress with ETA

### 🎨 UI/UX Features
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Dark/Light Mode**: Theme switching capability
- **Smooth Animations**: Framer Motion animations for better UX
- **Accessibility**: WCAG 2.1 AA compliant components

## Technology Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Charts**: Recharts
- **Icons**: Heroicons
- **State Management**: React Hooks + Local Storage
- **API Client**: Fetch with TypeScript types

## Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Backend API running on `http://localhost:8000`

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   # or
   yarn install
   ```

2. **Set up environment variables**:
   Create a `.env.local` file in the frontend directory:
   ```bash
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. **Open your browser**:
   Navigate to `http://localhost:3000`

### Building for Production

```bash
npm run build
npm start
```

## Project Structure

```
src/
├── app/                    # Next.js App Router
│   ├── dashboard/         # Dashboard page
│   ├── layout.tsx         # Root layout
│   ├── page.tsx          # Landing page
│   └── globals.css       # Global styles
├── components/            # React components
│   ├── dashboard/        # Dashboard-specific components
│   │   ├── dashboard.tsx
│   │   ├── dataset-upload.tsx
│   │   ├── pipeline-config.tsx
│   │   ├── pipeline-execution.tsx
│   │   └── results-visualization.tsx
│   └── ui/               # Reusable UI components
│       ├── alert.tsx
│       ├── loading-spinner.tsx
│       └── logo.tsx
└── lib/                  # Utility libraries
    ├── api.ts           # API client
    ├── config.ts        # Configuration
    ├── hooks.ts         # Custom React hooks
    └── types.ts         # TypeScript type definitions
```

## API Integration

The frontend communicates with the backend through a comprehensive API client that handles:

- **Dataset Analysis**: Upload and analyze datasets
- **Pipeline Execution**: Start, monitor, and manage ML pipelines
- **Real-time Updates**: Polling-based status updates
- **Results Retrieval**: Download and display results
- **Error Handling**: Comprehensive error management

### Key API Endpoints

- `POST /analyze` - Analyze uploaded dataset
- `POST /pipeline/run` - Start pipeline execution
- `GET /pipeline/status/{task_id}` - Get task status
- `POST /predict` - Make predictions
- `GET /model/info/{task_id}` - Get model information
- `GET /tasks` - List all tasks

## Customization

### Themes
The app supports light and dark themes. Theme configuration can be found in `src/lib/config.ts`.

### Configuration
All configuration options are centralized in `src/lib/config.ts`:
- API endpoints
- Feature flags
- Default pipeline settings
- UI preferences

### Adding New Components
1. Create component in appropriate directory
2. Export from component index if needed
3. Add TypeScript types in `src/lib/types.ts`
4. Update documentation

## Development Guidelines

### Code Style
- Use TypeScript for all components
- Follow React best practices
- Use Tailwind CSS for styling
- Implement proper error handling
- Write accessible components

### State Management
- Use React hooks for local state
- Use custom hooks for API interactions
- Store user preferences in localStorage
- Implement proper loading and error states

### Performance
- Implement proper loading states
- Use React.memo for expensive components
- Lazy load heavy components
- Optimize bundle size

## Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

## Deployment

### Vercel (Recommended)
1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### Docker
```bash
# Build Docker image
docker build -t ml-agent-frontend .

# Run container
docker run -p 3000:3000 ml-agent-frontend
```

### Static Export
```bash
npm run build
npm run export
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the API documentation

## Roadmap

### Upcoming Features
- [ ] Advanced data visualization
- [ ] Model comparison tools
- [ ] Automated report generation
- [ ] Team collaboration features
- [ ] API key management
- [ ] Advanced deployment options