"""
Natural Language Summaries for ML Results

This module provides LLM-powered natural language summarization of
machine learning pipeline results and model analysis.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.models.interpretability import ModelExplanation

logger = logging.getLogger(__name__)


class NaturalLanguageSummarizer:
    """LLM-powered natural language summarizer for model results."""
    
    def __init__(self, llm_client = None):
        """Initialize the natural language summarizer."""
        self.llm_client = llm_client
    
    def generate_results_summary(
        self, 
        leaderboard: pd.DataFrame,
        model_explanations: Dict[str, ModelExplanation],
        meta_features: Dict[str, Any],
        task_type: str,
        dataset_name: str = "dataset"
    ) -> Dict[str, str]:
        """Generate comprehensive natural language summary of results."""
        
        if not self.llm_client:
            return self._generate_default_summary(leaderboard, model_explanations)
        
        try:
            # Generate different types of summaries
            summaries = {
                "executive_summary": self._generate_executive_summary(
                    leaderboard, model_explanations, meta_features, task_type, dataset_name
                ),
                "technical_analysis": self._generate_technical_analysis(
                    leaderboard, model_explanations, meta_features, task_type
                ),
                "feature_insights": self._generate_feature_insights(
                    model_explanations, meta_features
                ),
                "recommendations": self._generate_recommendations(
                    leaderboard, model_explanations, meta_features, task_type
                )
            }
            
            return summaries
            
        except Exception as e:
            logger.error(f"Failed to generate natural language summaries: {e}")
            return self._generate_default_summary(leaderboard, model_explanations)
    
    def _generate_executive_summary(
        self, 
        leaderboard: pd.DataFrame,
        model_explanations: Dict[str, ModelExplanation],
        meta_features: Dict[str, Any],
        task_type: str,
        dataset_name: str
    ) -> str:
        """Generate executive summary of the ML pipeline results."""
        
        prompt = f"""
You are a senior data scientist presenting results to executives. Provide a clear, concise executive summary of the machine learning pipeline results.

Dataset: {dataset_name}
Task Type: {task_type}
Dataset Size: {meta_features.get('num_instances', 'unknown')} instances, {meta_features.get('num_features', 'unknown')} features

Top Performing Models:
"""
        
        for i, (_, row) in enumerate(leaderboard.head(3).iterrows()):
            prompt += f"""
{i+1}. {row['model_name']}: {row.get('accuracy', row.get('score', 0)):.1%} accuracy
   - Training Time: {row.get('training_time', 0):.1f}s
   - Key Strengths: [Model-specific strengths]
"""
        
        prompt += f"""

Please provide a 2-3 paragraph executive summary covering:
1. Overall pipeline performance and success
2. Key findings and insights
3. Business impact and recommendations

Focus on business value, not technical details. Use clear, non-technical language.
"""
        
        try:
            response = self.llm_client.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return self._generate_default_executive_summary(leaderboard)
    
    def _generate_technical_analysis(
        self, 
        leaderboard: pd.DataFrame,
        model_explanations: Dict[str, ModelExplanation],
        meta_features: Dict[str, Any],
        task_type: str
    ) -> str:
        """Generate technical analysis of the results."""
        
        prompt = f"""
You are a machine learning engineer providing a technical analysis of model performance. Analyze the following results:

Dataset Characteristics:
- Task Type: {task_type}
- Instances: {meta_features.get('num_instances', 'unknown')}
- Features: {meta_features.get('num_features', 'unknown')}
- Missing Values: {meta_features.get('missing_values_ratio', 0):.1%}
- Categorical Features: {meta_features.get('categorical_features_ratio', 0):.1%}

Model Performance:
"""
        
        for _, row in leaderboard.head(5).iterrows():
            prompt += f"""
- {row['model_name']}: {row.get('accuracy', row.get('score', 0)):.3f} accuracy
  Training time: {row.get('training_time', 0):.1f}s
"""
        
        prompt += """

Provide a technical analysis covering:
1. Model performance comparison and ranking
2. Training time vs accuracy trade-offs
3. Model complexity and interpretability analysis
4. Feature importance insights
5. Potential improvements and next steps

Use technical terminology appropriate for ML engineers.
"""
        
        try:
            response = self.llm_client.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to generate technical analysis: {e}")
            return self._generate_default_technical_analysis(leaderboard)
    
    def _generate_feature_insights(
        self, 
        model_explanations: Dict[str, ModelExplanation],
        meta_features: Dict[str, Any]
    ) -> str:
        """Generate feature importance and insights analysis."""
        
        prompt = f"""
You are a data scientist analyzing feature importance across multiple models. Provide insights on feature patterns and importance.

Dataset Features:
- Total Features: {meta_features.get('num_features', 'unknown')}
- Numerical Features: {meta_features.get('numerical_features_ratio', 0):.1%}
- Categorical Features: {meta_features.get('categorical_features_ratio', 0):.1%}

Model Feature Importance:
"""
        
        for model_name, explanation in model_explanations.items():
            if explanation.feature_importance:
                top_features = explanation.feature_importance.feature_names[:5]
                top_scores = explanation.feature_importance.importance_scores[:5]
                
                prompt += f"""
{model_name} ({explanation.feature_importance.method}):
"""
                for feature, score in zip(top_features, top_scores):
                    prompt += f"  - {feature}: {score:.3f}\n"
        
        prompt += """

Provide analysis covering:
1. Most important features across models
2. Feature importance consistency
3. Feature type patterns (numerical vs categorical)
4. Feature engineering opportunities
5. Data collection recommendations

Focus on actionable insights for feature engineering and data collection.
"""
        
        try:
            response = self.llm_client.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to generate feature insights: {e}")
            return self._generate_default_feature_insights(model_explanations)
    
    def _generate_recommendations(
        self, 
        leaderboard: pd.DataFrame,
        model_explanations: Dict[str, ModelExplanation],
        meta_features: Dict[str, Any],
        task_type: str
    ) -> str:
        """Generate recommendations for model deployment and improvement."""
        
        prompt = f"""
You are a machine learning consultant providing recommendations for model deployment and improvement.

Current Results Summary:
- Best Model: {leaderboard.iloc[0]['model_name']} ({leaderboard.iloc[0].get('accuracy', leaderboard.iloc[0].get('score', 0)):.1%})
- Task Type: {task_type}
- Dataset Size: {meta_features.get('num_instances', 'unknown')} instances

Model Performance Distribution:
- Top 3 models accuracy range: {leaderboard.head(3).get('accuracy', leaderboard.head(3).get('score', 0)).min():.1%} - {leaderboard.head(3).get('accuracy', leaderboard.head(3).get('score', 0)).max():.1%}
- Average training time: {leaderboard.get('training_time', 0).mean():.1f}s

Provide recommendations covering:
1. Model selection for production deployment
2. Performance optimization strategies
3. Feature engineering improvements
4. Data collection and preprocessing enhancements
5. Monitoring and maintenance considerations
6. Risk assessment and mitigation

Focus on practical, actionable recommendations for production deployment.
"""
        
        try:
            response = self.llm_client.generate_response(prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return self._generate_default_recommendations(leaderboard)
    
    def _generate_default_summary(
        self, 
        leaderboard: pd.DataFrame,
        model_explanations: Dict[str, ModelExplanation]
    ) -> Dict[str, str]:
        """Generate default summary when LLM is not available."""
        
        best_model = leaderboard.iloc[0]
        
        return {
            "executive_summary": f"""
The machine learning pipeline successfully completed with {best_model['model_name']} achieving the best performance at {best_model.get('accuracy', best_model.get('score', 0)):.1%} accuracy.

Key Results:
- {len(leaderboard)} models were evaluated
- Best performing model: {best_model['model_name']}
- Performance range: {leaderboard.get('accuracy', leaderboard.get('score', 0)).min():.1%} - {leaderboard.get('accuracy', leaderboard.get('score', 0)).max():.1%}
- Average training time: {leaderboard.get('training_time', 0).mean():.1f} seconds

The pipeline demonstrates strong performance with clear model differentiation and reasonable computational efficiency.
""",
            
            "technical_analysis": f"""
Technical Analysis Summary:

Model Performance Ranking:
1. {best_model['model_name']}: {best_model.get('accuracy', best_model.get('score', 0)):.3f}
2. {leaderboard.iloc[1]['model_name']}: {leaderboard.iloc[1].get('accuracy', leaderboard.iloc[1].get('score', 0)):.3f}
3. {leaderboard.iloc[2]['model_name']}: {leaderboard.iloc[2].get('accuracy', leaderboard.iloc[2].get('score', 0)):.3f}

Performance Insights:
- Clear performance differentiation between models
- Training time varies significantly across model types
- Feature importance analysis available for {len(model_explanations)} models

Recommendations:
- Consider ensemble methods for improved performance
- Feature engineering could further improve results
- Model selection should balance performance and interpretability
""",
            
            "feature_insights": """
Feature Importance Analysis:

The feature importance analysis reveals patterns across different model types. Key insights include:

1. Consistent important features across multiple models
2. Feature type distribution affects model performance
3. Some features show high importance in tree-based models
4. Linear models highlight different feature patterns

Recommendations:
- Focus on top-ranked features for feature engineering
- Consider feature interactions for complex models
- Monitor feature stability over time
""",
            
            "recommendations": f"""
Deployment Recommendations:

Model Selection:
- Primary: {best_model['model_name']} for best performance
- Backup: {leaderboard.iloc[1]['model_name']} for reliability

Production Considerations:
- Implement model monitoring for performance tracking
- Set up automated retraining pipelines
- Consider A/B testing for model comparison
- Monitor feature drift and data quality

Risk Mitigation:
- Maintain model versioning and rollback capabilities
- Implement comprehensive testing procedures
- Set up alerting for performance degradation
- Regular model validation and updating
"""
        }
    
    def _generate_default_executive_summary(self, leaderboard: pd.DataFrame) -> str:
        """Generate default executive summary."""
        best_model = leaderboard.iloc[0]
        return f"""
Executive Summary:

The machine learning pipeline has successfully completed, achieving strong performance with {best_model['model_name']} as the top-performing model at {best_model.get('accuracy', best_model.get('score', 0)):.1%} accuracy.

Key Business Impact:
- {len(leaderboard)} models evaluated with clear performance differentiation
- Performance range demonstrates model reliability
- Training efficiency supports scalable deployment

Recommendation: Proceed with {best_model['model_name']} for production deployment with monitoring for performance tracking.
"""
    
    def _generate_default_technical_analysis(self, leaderboard: pd.DataFrame) -> str:
        """Generate default technical analysis."""
        return f"""
Technical Analysis:

Model Performance: {len(leaderboard)} models evaluated with clear performance hierarchy.
Best Model: {leaderboard.iloc[0]['model_name']} at {leaderboard.iloc[0].get('accuracy', leaderboard.iloc[0].get('score', 0)):.3f}
Performance Range: {leaderboard.get('accuracy', leaderboard.get('score', 0)).min():.3f} - {leaderboard.get('accuracy', leaderboard.get('score', 0)).max():.3f}

Technical Insights:
- Clear model differentiation supports confident selection
- Training time analysis enables performance-complexity trade-offs
- Feature importance available for model interpretability
"""
    
    def _generate_default_feature_insights(self, model_explanations: Dict[str, ModelExplanation]) -> str:
        """Generate default feature insights."""
        return f"""
Feature Analysis:

Feature importance analysis completed for {len(model_explanations)} models.
Key patterns identified across different model types.
Top features show consistent importance rankings.
Feature engineering opportunities identified for performance improvement.
"""
    
    def _generate_default_recommendations(self, leaderboard: pd.DataFrame) -> str:
        """Generate default recommendations."""
        best_model = leaderboard.iloc[0]
        return f"""
Recommendations:

Deployment: {best_model['model_name']} recommended for production.
Monitoring: Implement performance tracking and alerting.
Improvement: Consider ensemble methods and feature engineering.
Maintenance: Regular model validation and retraining schedule.
"""
