"""
Leaderboard for tracking model performance
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np


class Leaderboard:
    """Leaderboard to track and rank model performance"""
    
    def __init__(self):
        self.results = []
        self.leaderboard_df = pd.DataFrame()
    
    def update_results(self, results: List[Dict[str, Any]]):
        """Update leaderboard with new results"""
        self.results = results
        self._update_dataframe()
    
    def _update_dataframe(self):
        """Update the leaderboard DataFrame"""
        if not self.results:
            self.leaderboard_df = pd.DataFrame()
            return
        
        # Convert results to DataFrame
        df_data = []
        for result in self.results:
            row = {
                'model_name': result.get('model_name', 'Unknown'),
                'accuracy': result.get('accuracy', 0.0),
                'precision': result.get('precision', 0.0),
                'recall': result.get('recall', 0.0),
                'f1_score': result.get('f1_score', 0.0),
                'training_time': result.get('training_time', 0.0),
                'cross_val_score': result.get('cross_val_score', 0.0),
                'cross_val_std': result.get('cross_val_std', 0.0)
            }
            df_data.append(row)
        
        self.leaderboard_df = pd.DataFrame(df_data)
        
        # Sort by accuracy (or primary metric)
        if not self.leaderboard_df.empty:
            self.leaderboard_df = self.leaderboard_df.sort_values('accuracy', ascending=False)
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get the current leaderboard"""
        return self.leaderboard_df.copy()
    
    def get_top_models(self, k: int = 3) -> List[Any]:
        """Get top k models from leaderboard"""
        if self.leaderboard_df.empty:
            return []
        
        top_models = []
        for i in range(min(k, len(self.leaderboard_df))):
            model_name = self.leaderboard_df.iloc[i]['model_name']
            # Find the actual model object from results
            for result in self.results:
                if result.get('model_name') == model_name:
                    top_models.append(result.get('model'))
                    break
        
        return top_models
    
    def get_best_model(self) -> Optional[Any]:
        """Get the best performing model"""
        top_models = self.get_top_models(1)
        return top_models[0] if top_models else None
    
    def get_best_score(self) -> float:
        """Get the best score"""
        if self.leaderboard_df.empty:
            return 0.0
        return self.leaderboard_df.iloc[0]['accuracy']
    
    def add_result(self, result: Dict[str, Any]):
        """Add a single result to the leaderboard"""
        self.results.append(result)
        self._update_dataframe()
    
    def clear(self):
        """Clear the leaderboard"""
        self.results = []
        self.leaderboard_df = pd.DataFrame()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the leaderboard"""
        if self.leaderboard_df.empty:
            return {}
        
        return {
            'total_models': len(self.leaderboard_df),
            'best_accuracy': self.leaderboard_df['accuracy'].max(),
            'worst_accuracy': self.leaderboard_df['accuracy'].min(),
            'mean_accuracy': self.leaderboard_df['accuracy'].mean(),
            'std_accuracy': self.leaderboard_df['accuracy'].std(),
            'total_training_time': self.leaderboard_df['training_time'].sum(),
            'mean_training_time': self.leaderboard_df['training_time'].mean()
        }
