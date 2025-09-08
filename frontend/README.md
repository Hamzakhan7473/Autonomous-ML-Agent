# Autonomous ML Agent - Frontend

A modern, responsive React/Next.js frontend for the Autonomous Machine Learning Agent. This application provides an intuitive interface for uploading datasets, configuring ML pipelines, and visualizing results.

## 🚀 Features

### 🎨 Modern Design
- **Beautiful UI**: Clean, modern interface with smooth animations
- **Dark/Light Mode**: Toggle between themes with smooth transitions
- **Responsive Design**: Optimized for mobile, tablet, and desktop
- **Accessibility**: Built with accessibility best practices

### 📊 Dashboard Features
- **Data Upload**: Drag-and-drop CSV file upload with validation
- **Pipeline Configuration**: Intuitive parameter setting with real-time feedback
- **Live Execution**: Real-time pipeline progress with step-by-step tracking
- **Results Visualization**: Interactive charts and model performance metrics
- **AI Insights**: Generated explanations and recommendations

### 🛠 Technical Features
- **TypeScript**: Full type safety and better developer experience
- **Tailwind CSS**: Utility-first CSS framework for rapid styling
- **Framer Motion**: Smooth animations and transitions
- **Radix UI**: Accessible, unstyled UI components
- **Heroicons**: Beautiful SVG icons

## 🏗 Architecture

```
src/
├── app/                    # Next.js app router
│   ├── globals.css        # Global styles and CSS variables
│   ├── layout.tsx         # Root layout with metadata
│   └── page.tsx           # Main page component
├── components/
│   ├── dashboard/         # Dashboard-specific components
│   │   ├── data-upload.tsx
│   │   ├── pipeline-config.tsx
│   │   ├── pipeline-execution.tsx
│   │   └── results-visualization.tsx
│   ├── landing/           # Landing page components
│   │   └── hero.tsx
│   ├── layout/            # Layout components
│   │   ├── header.tsx
│   │   ├── navigation.tsx
│   │   ├── sidebar.tsx
│   │   └── mobile-step-indicator.tsx
│   └── ui/                # Reusable UI components
│       ├── badge.tsx
│       ├── button.tsx
│       ├── card.tsx
│       └── progress.tsx
└── lib/
    └── utils.ts           # Utility functions
```

## 🎯 Key Components

### Landing Page
- **Hero Section**: Compelling introduction with feature highlights
- **Navigation**: Responsive navigation with mobile menu
- **Call-to-Action**: Clear path to get started

### Dashboard
- **Step-by-Step Workflow**: Guided pipeline creation process
- **Real-Time Updates**: Live progress tracking during execution
- **Interactive Results**: Comprehensive model performance visualization
- **Mobile-Friendly**: Optimized for all screen sizes

### UI Components
- **Design System**: Consistent, reusable components
- **Accessibility**: WCAG compliant components
- **Animations**: Smooth, purposeful motion design

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Open in browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## 🎨 Design System

### Colors
- **Primary**: Blue gradient (`from-blue-600 to-purple-600`)
- **Secondary**: Muted grays for subtle elements
- **Success**: Green for positive states
- **Warning**: Yellow for caution states
- **Destructive**: Red for error states

### Typography
- **Font**: Geist Sans (primary), Geist Mono (code)
- **Scale**: Consistent sizing with Tailwind's typography scale
- **Hierarchy**: Clear visual hierarchy with proper contrast

### Spacing
- **Consistent**: 4px base unit with Tailwind spacing scale
- **Responsive**: Adaptive spacing for different screen sizes
- **Breathing Room**: Adequate whitespace for readability

### Components
- **Cards**: Elevated containers with subtle shadows
- **Buttons**: Multiple variants with hover states
- **Forms**: Accessible form controls with validation
- **Progress**: Visual progress indicators
- **Badges**: Status and category indicators

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px  
- **Desktop**: > 1024px

### Mobile Optimizations
- **Touch-Friendly**: Larger touch targets
- **Simplified Navigation**: Collapsible mobile menu
- **Step Indicator**: Horizontal progress indicator
- **Optimized Layout**: Stacked layouts for small screens

## 🌙 Dark Mode

### Implementation
- **CSS Variables**: Dynamic theming with CSS custom properties
- **Smooth Transitions**: Animated theme switching
- **System Preference**: Respects user's system preference
- **Manual Toggle**: User-controlled theme switching

### Color Scheme
- **Light**: Clean whites and subtle grays
- **Dark**: Deep grays with high contrast accents
- **Consistent**: Maintains brand colors in both themes

## 🎭 Animations

### Framer Motion
- **Page Transitions**: Smooth route transitions
- **Component Animations**: Entrance and exit animations
- **Micro-Interactions**: Button hover and click effects
- **Loading States**: Animated progress indicators

### Performance
- **Optimized**: Hardware-accelerated animations
- **Reduced Motion**: Respects user preferences
- **Smooth**: 60fps animations with proper easing

## 🔧 Customization

### Theme Customization
Edit `src/app/globals.css` to modify:
- Color schemes
- Typography scales
- Spacing values
- Border radius

### Component Customization
All UI components are built with:
- **Variants**: Multiple style options
- **Composition**: Flexible component composition
- **Theming**: CSS variable-based theming

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

### Deploy to Vercel
```bash
npx vercel
```

### Environment Variables
- `NEXT_PUBLIC_API_URL` - Backend API URL
- `NEXT_PUBLIC_APP_NAME` - Application name

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- **TypeScript**: Use strict typing
- **Components**: Follow component composition patterns
- **Styling**: Use Tailwind utility classes
- **Accessibility**: Ensure WCAG compliance
- **Testing**: Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Next.js** - React framework
- **Tailwind CSS** - Utility-first CSS
- **Framer Motion** - Animation library
- **Radix UI** - Accessible components
- **Heroicons** - Beautiful icons