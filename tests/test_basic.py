"""Basic tests for the Autonomous ML Agent."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.core.ingest import analyze_data, DataIngester
from src.core.preprocess import DataPreprocessor, PreprocessingConfig
from src.core.model_zoo import model_zoo
from src.core.search import HyperparameterOptimizer
from src.core.evaluate import ModelEvaluator
from src.ensemble.blending import create_ensemble


class TestDataIngestion:
    """Test data ingestion functionality."""
    
    def test_data_ingester_initialization(self):
        """Test DataIngester initialization."""
        ingester = DataIngester()
        assert ingester.random_state == 42
    
    def test_create_sample_data(self):
        """Create sample data for testing."""
        # Create sample iris-like data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'sepal_length': np.random.normal(5.8, 0.8, n_samples),
            'sepal_width': np.random.normal(3.0, 0.4, n_samples),
            'petal_length': np.random.normal(3.8, 1.8, n_samples),
            'petal_width': np.random.normal(1.2, 0.8, n_samples),
            'species': np.random.choice(['setosa', 'versicolor', 'virginica'], n_samples)
        }
        
        df = pd.DataFrame(data)
        return df
    
    def test_analyze_data(self):
        """Test data analysis functionality."""
        df = self.test_create_sample_data()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Test analysis
            analyzed_df, schema, summary = analyze_data(temp_path, 'species')
            
            assert analyzed_df.shape == df.shape
            assert schema.n_rows == 100
            assert schema.n_features == 4
            assert schema.target_type == 'categorical'
            assert 'species' in summary['target_summary']
            
        finally:
            os.unlink(temp_path)


class TestPreprocessing:
    """Test preprocessing functionality."""
    
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)
        assert preprocessor.config == config
    
    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline."""
        # Create sample data with missing values
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Introduce missing values
        df.loc[0:10, 'feature1'] = np.nan
        df.loc[5:15, 'feature3'] = np.nan
        
        X = df.drop(columns=['target'])
        y = df['target']
        
        # Test preprocessing
        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)
        
        assert X_processed.shape[0] == X.shape[0]
        assert not X_processed.isna().any().any()


class TestModelZoo:
    """Test model zoo functionality."""
    
    def test_model_zoo_initialization(self):
        """Test model zoo initialization."""
        assert len(model_zoo.models) > 0
    
    def test_list_models(self):
        """Test model listing."""
        classification_models = model_zoo.list_models(is_classification=True)
        regression_models = model_zoo.list_models(is_classification=False)
        
        assert len(classification_models) > 0
        assert len(regression_models) > 0
        assert 'logistic_regression' in classification_models
        assert 'linear_regression' in regression_models
    
    def test_get_model(self):
        """Test model retrieval."""
        # Test classification model
        model = model_zoo.get_model('logistic_regression', is_classification=True)
        assert model.config.name == 'logistic_regression'
        assert model.config.is_classification is True
        
        # Test regression model
        model = model_zoo.get_model('linear_regression', is_classification=False)
        assert model.config.name == 'linear_regression'
        assert model.config.is_classification is False
    
    def test_model_training(self):
        """Test model training."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples)
        })
        
        y = pd.Series(np.random.choice([0, 1], n_samples))
        
        # Test logistic regression
        model = model_zoo.get_model('logistic_regression', is_classification=True)
        model.fit(X, y)
        
        assert model.is_fitted is True
        assert len(model.feature_names) == 3
        
        # Test predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        
        # Test probabilities
        probabilities = model.predict_proba(X)
        assert probabilities.shape[0] == len(y)
        assert probabilities.shape[1] == 2  # Binary classification


class TestHyperparameterOptimization:
    """Test hyperparameter optimization."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        model = model_zoo.get_model('logistic_regression', is_classification=True)
        optimizer = HyperparameterOptimizer(model, n_trials=5, timeout=60)
        
        assert optimizer.model == model
        assert optimizer.config.n_trials == 5
        assert optimizer.config.timeout == 60
    
    def test_random_search(self):
        """Test random search optimization."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        
        y = pd.Series(np.random.choice([0, 1], n_samples))
        
        # Test optimization
        model = model_zoo.get_model('logistic_regression', is_classification=True)
        optimizer = HyperparameterOptimizer(model, method='random', n_trials=5, timeout=60)
        
        best_model, best_score, best_params = optimizer.optimize(X, y)
        
        assert best_model is not None
        assert best_score > 0
        assert best_params is not None


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator.config.cv_folds == 5
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        
        y = pd.Series(np.random.choice([0, 1], n_samples))
        
        # Train model
        model = model_zoo.get_model('logistic_regression', is_classification=True)
        model.fit(X, y)
        
        # Evaluate model
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(model, X, y)
        
        assert 'cv_results' in results
        assert 'detailed_results' in results
        assert 'primary_metric' in results
        assert results['primary_metric'] == 'accuracy'


class TestEnsemble:
    """Test ensemble functionality."""
    
    def test_ensemble_creation(self):
        """Test ensemble creation."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        
        y = pd.Series(np.random.choice([0, 1], n_samples))
        
        # Train multiple models
        models = []
        for model_name in ['logistic_regression', 'random_forest']:
            model = model_zoo.get_model(model_name, is_classification=True)
            model.fit(X, y)
            models.append(model)
        
        # Create ensemble
        ensemble = create_ensemble(models, X, y, method='weighted')
        
        assert ensemble is not None
        assert ensemble.is_fitted is True
        
        # Test predictions
        predictions = ensemble.predict(X)
        assert len(predictions) == len(y)


if __name__ == "__main__":
    pytest.main([__file__])
