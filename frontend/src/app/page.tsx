"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { 
  SparklesIcon, 
  ChartBarIcon, 
  CpuChipIcon,
  RocketLaunchIcon,
  ArrowRightIcon,
  PlayIcon
} from "@heroicons/react/24/outline";
import Logo from "@/components/ui/logo";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-blue-900">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200 dark:border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <Logo size="md" />

            {/* Navigation Links */}
            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white transition-colors">
                Features
              </a>
              <a href="#how-it-works" className="text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white transition-colors">
                How it Works
              </a>
              <a href="#pricing" className="text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white transition-colors">
                Pricing
              </a>
            </div>

            {/* CTA Button */}
            <a 
              href="/dashboard"
              className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-2 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 font-medium"
            >
              Get Started
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-full border border-blue-200 dark:border-blue-800 mb-8">
              <Logo size="sm" showText={false} className="mr-3" />
              <span className="text-blue-700 dark:text-blue-300 text-sm font-semibold">
                AI-Powered Machine Learning Platform
              </span>
            </div>

            <h1 className="text-5xl md:text-7xl font-bold text-slate-900 dark:text-white mb-6 leading-tight">
              Build Better Models
              <br />
              <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Automatically
              </span>
            </h1>

            <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto mb-12 leading-relaxed">
              Upload your data, configure your pipeline, and let our AI agent automatically build, 
              train, and optimize machine learning models for you. No coding required.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
              <motion.a
                href="/dashboard"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-lg text-lg font-medium hover:from-blue-700 hover:to-purple-700 transition-all duration-200 flex items-center justify-center"
              >
                Start Building
                <ArrowRightIcon className="w-5 h-5 ml-2" />
              </motion.a>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300 px-8 py-4 rounded-lg text-lg font-medium hover:bg-slate-50 dark:hover:bg-slate-800 transition-all duration-200 flex items-center justify-center"
              >
                <PlayIcon className="w-5 h-5 mr-2" />
                Watch Demo
              </motion.button>
            </div>
          </motion.div>

          {/* Feature Cards */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto"
          >
            <div className="bg-white dark:bg-slate-800 p-8 rounded-2xl shadow-lg hover:shadow-xl transition-shadow duration-300">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center mb-6 mx-auto">
                <CpuChipIcon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
                AI-Powered Pipeline
              </h3>
              <p className="text-slate-600 dark:text-slate-300">
                Intelligent preprocessing, feature engineering, and model selection powered by advanced AI algorithms.
              </p>
            </div>

            <div className="bg-white dark:bg-slate-800 p-8 rounded-2xl shadow-lg hover:shadow-xl transition-shadow duration-300">
              <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-blue-600 rounded-xl flex items-center justify-center mb-6 mx-auto">
                <ChartBarIcon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
                Automated Optimization
              </h3>
              <p className="text-slate-600 dark:text-slate-300">
                Hyperparameter tuning and model optimization with meta-learning for faster convergence.
              </p>
            </div>

            <div className="bg-white dark:bg-slate-800 p-8 rounded-2xl shadow-lg hover:shadow-xl transition-shadow duration-300">
              <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center mb-6 mx-auto">
                <RocketLaunchIcon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
                Production Ready
              </h3>
              <p className="text-slate-600 dark:text-slate-300">
                Deploy models with confidence using our automated deployment pipeline and monitoring.
              </p>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
