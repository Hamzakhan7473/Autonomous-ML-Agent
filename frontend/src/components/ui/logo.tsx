'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  animated?: boolean;
  className?: string;
}

const Logo: React.FC<LogoProps> = ({ 
  size = 'md', 
  animated = true, 
  className = '' 
}) => {
  const sizeClasses = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16',
    xl: 'w-20 h-20'
  };

  return (
    <motion.div
      className={`relative ${sizeClasses[size]} ${className}`}
      animate={animated ? {
        scale: [1, 1.02, 1]
      } : {}}
      transition={{
        duration: 6,
        repeat: Infinity,
        repeatDelay: 3,
        ease: "easeInOut"
      }}
    >
      {/* Main Logo Container */}
      <div className="relative w-full h-full">
        {/* Background Circle with Gradient */}
        <div 
          className="absolute inset-0 rounded-full"
          style={{
            background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)',
            boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.3), 0 4px 16px rgba(6,182,212,0.2)'
          }}
        />
        
        {/* Elegant Border */}
        <motion.div
          className="absolute inset-0 rounded-full border border-white/20"
          animate={animated ? {
            borderColor: ['rgba(255,255,255,0.2)', 'rgba(6,182,212,0.6)', 'rgba(255,255,255,0.2)']
          } : {}}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        
        {/* Central Design - Abstract "A" for Autonomous */}
        <div className="absolute inset-0 flex items-center justify-center">
          <svg
            className="w-3/4 h-3/4"
            viewBox="0 0 100 100"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Main "A" Structure */}
            <motion.path
              d="M25 75 L50 25 L75 75 M35 60 L65 60"
              stroke="url(#logoGradient)"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
              animate={animated ? {
                strokeDasharray: ['0 100', '100 0'],
                strokeDashoffset: [0, 100]
              } : {}}
              transition={{
                duration: 2,
                repeat: Infinity,
                repeatDelay: 1,
                ease: "easeInOut"
              }}
            />
            
            {/* Subtle accent lines */}
            <motion.path
              d="M20 70 Q50 20 80 70"
              stroke="url(#accentGradient)"
              strokeWidth="1.5"
              strokeLinecap="round"
              fill="none"
              opacity="0.4"
              animate={animated ? {
                pathLength: [0, 1, 0]
              } : {}}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 0.5
              }}
            />
            
            {/* Central dot */}
            <motion.circle
              cx="50"
              cy="50"
              r="3"
              fill="url(#logoGradient)"
              animate={animated ? {
                scale: [1, 1.3, 1],
                opacity: [0.8, 1, 0.8]
              } : {}}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
            
            {/* Gradients */}
            <defs>
              <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" />
                <stop offset="50%" stopColor="#8b5cf6" />
                <stop offset="100%" stopColor="#ec4899" />
              </linearGradient>
              <linearGradient id="accentGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.6" />
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.6" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        
        {/* Subtle glow effect */}
        <motion.div
          className="absolute inset-0 rounded-full opacity-0"
          style={{
            background: 'radial-gradient(circle, rgba(6,182,212,0.1) 0%, rgba(139,92,246,0.05) 50%, transparent 100%)'
          }}
          animate={animated ? {
            opacity: [0, 0.8, 0],
            scale: [1, 1.1, 1]
          } : {}}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </div>
    </motion.div>
  );
};

export default Logo;