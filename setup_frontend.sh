#!/bin/bash

# Frontend Setup Script for Autonomous ML Agent

echo "🎨 Setting up Autonomous ML Agent Frontend..."

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    echo "❌ Error: frontend directory not found. Please run this script from the project root."
    exit 1
fi

# Navigate to frontend directory
cd frontend

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed. Please install Node.js 18+ and try again."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Error: Node.js version 18+ is required. Current version: $(node --version)"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed."
    exit 1
fi

echo "✅ npm version: $(npm --version)"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies."
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Create environment file
echo "🔧 Setting up environment configuration..."
if [ ! -f ".env.local" ]; then
    cat > .env.local << EOF
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Development Configuration
NODE_ENV=development
EOF
    echo "✅ Created .env.local file"
else
    echo "⚠️  .env.local already exists, skipping creation"
fi

# Run type checking
echo "🔍 Running type checking..."
npm run type-check

if [ $? -ne 0 ]; then
    echo "⚠️  Type checking found issues. Please review and fix them."
else
    echo "✅ Type checking passed"
fi

# Run linting
echo "🧹 Running linting..."
npm run lint

if [ $? -ne 0 ]; then
    echo "⚠️  Linting found issues. You can fix them with: npm run lint:fix"
else
    echo "✅ Linting passed"
fi

# Build the project
echo "🏗️  Building the project..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Error: Build failed. Please check the errors above."
    exit 1
fi

echo "✅ Build successful"

echo ""
echo "🎉 Frontend setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Start the backend API server:"
echo "   cd .. && python -m src.service.app"
echo ""
echo "2. Start the frontend development server:"
echo "   npm run dev"
echo ""
echo "3. Open your browser and navigate to:"
echo "   http://localhost:3000"
echo ""
echo "🔧 Available commands:"
echo "   npm run dev          - Start development server"
echo "   npm run build        - Build for production"
echo "   npm run start        - Start production server"
echo "   npm run lint         - Run linting"
echo "   npm run lint:fix     - Fix linting issues"
echo "   npm run type-check   - Run TypeScript type checking"
echo ""
echo "📚 Documentation:"
echo "   - Frontend README: frontend/README.md"
echo "   - API Documentation: http://localhost:8000/docs"
echo ""
echo "🚀 Happy coding!"
