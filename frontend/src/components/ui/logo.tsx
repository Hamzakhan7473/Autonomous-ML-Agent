import React from 'react';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg';
  showText?: boolean;
  className?: string;
}

export function Logo({ size = 'md', showText = true, className = '' }: LogoProps) {
  const sizeClasses = {
    sm: 'w-6 h-6 text-xs',
    md: 'w-8 h-8 text-sm',
    lg: 'w-12 h-12 text-lg'
  };

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-lg',
    lg: 'text-2xl'
  };

  return (
    <div className={`flex items-center space-x-3 ${className}`}>
      {/* Logo Icon */}
      <div className={`${sizeClasses[size]} bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center shadow-lg font-bold text-white`}>
        ML
      </div>
      
      {/* Logo Text */}
      {showText && (
        <div className="flex flex-col">
          <span className={`${textSizeClasses[size]} font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent`}>
            Autonomous ML Agent
          </span>
          {size === 'lg' && (
            <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">
              AI-Powered Machine Learning Platform
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default Logo;

