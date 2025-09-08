"""Command-line interface for the Autonomous ML Agent."""

import click
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import time
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

from .core.ingest import analyze_data
from .core.preprocess import DataPreprocessor, PreprocessingConfig
from .core.model_zoo import model_zoo
from .core.search import HyperparameterOptimizer
from .core.evaluate import ModelEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()


@click.group()
@click.version_option()
def cli():
    """🤖 Autonomous Machine Learning Agent CLI
    
    An intelligent, LLM-orchestrated machine learning pipeline that automatically
    ingests tabular datasets, cleans and preprocesses data, trains multiple models,
    and optimizes them for target metrics.
    """
    pass


@cli.command()
@click.option('--data', '-d', required=True, help='Path to the dataset file')
@click.option('--target', '-t', required=True, help='Name of the target column')
@click.option('--metric', '-m', default='auto', help='Optimization metric (accuracy, f1, auc, mse, mae, auto)')
@click.option('--budget-minutes', default=30, help='Time budget in minutes')
@click.option('--time-limit', default=1800, help='Time limit in seconds')
@click.option('--seed', default=42, help='Random seed for reproducibility')
@click.option('--models', help='Comma-separated list of models to try')
@click.option('--output-dir', '-o', default='./results', help='Output directory for results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def run(data: str, target: str, metric: str, budget_minutes: int, time_limit: int, 
        seed: int, models: Optional[str], output_dir: str, verbose: bool):
    """Run the complete ML pipeline on a dataset."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel.fit(
        f"[bold blue]🤖 Autonomous ML Agent[/bold blue]\n"
        f"Dataset: {data}\n"
        f"Target: {target}\n"
        f"Metric: {metric}\n"
        f"Time Budget: {budget_minutes} minutes\n"
        f"Random Seed: {seed}",
        title="Pipeline Configuration"
    ))
    
    try:
        # Step 1: Data Analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing dataset...", total=None)
            
            df, schema, summary = analyze_data(data, target)
            
            progress.update(task, description="✅ Dataset analysis complete")
        
        # Display data summary
        _display_data_summary(schema, summary)
        
        # Determine task type
        is_classification = schema.target_type == 'categorical'
        
        # Select models
        if models:
            model_names = [m.strip() for m in models.split(',')]
        else:
            model_names = model_zoo.get_recommended_models(
                is_classification, schema.n_rows, schema.n_features
            )
        
        console.print(f"\n[bold green]Selected models:[/bold green] {', '.join(model_names)}")
        
        # Step 2: Preprocessing
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Preprocessing data...", total=None)
            
            preprocessor = DataPreprocessor()
            X = df.drop(columns=[target])
            y = df[target]
            
            X_processed = preprocessor.fit_transform(X, y)
            
            progress.update(task, description="✅ Data preprocessing complete")
        
        # Step 3: Model Training and Optimization
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for i, model_name in enumerate(model_names):
                task = progress.add_task(f"Training {model_name}...", total=None)
                
                try:
                    # Get model
                    model = model_zoo.get_model(model_name, is_classification)
                    
                    # Hyperparameter optimization
                    optimizer = HyperparameterOptimizer(
                        model=model,
                        cv_folds=5,
                        n_trials=20,
                        timeout=time_limit // len(model_names)
                    )
                    
                    best_model, best_score, best_params = optimizer.optimize(X_processed, y)
                    
                    # Evaluate model
                    evaluator = ModelEvaluator()
                    metrics = evaluator.evaluate(best_model, X_processed, y, cv_folds=5)
                    
                    results.append({
                        'model': model_name,
                        'best_score': best_score,
                        'best_params': best_params,
                        'metrics': metrics,
                        'training_time': best_model.training_time
                    })
                    
                    progress.update(task, description=f"✅ {model_name} complete (score: {best_score:.4f})")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    progress.update(task, description=f"❌ {model_name} failed")
        
        # Step 4: Display Results
        _display_results(results, metric)
        
        # Step 5: Save Results
        _save_results(results, output_path, schema, summary)
        
        console.print(f"\n[bold green]✅ Pipeline complete! Results saved to {output_path}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.Abort()


@cli.command()
@click.option('--data', '-d', required=True, help='Path to the dataset file')
@click.option('--target', '-t', required=True, help='Name of the target column')
def analyze(data: str, target: str):
    """Analyze a dataset and display detailed information."""
    
    console.print(Panel.fit(
        f"[bold blue]📊 Dataset Analysis[/bold blue]\n"
        f"Dataset: {data}\n"
        f"Target: {target}",
        title="Analysis Configuration"
    ))
    
    try:
        df, schema, summary = analyze_data(data, target)
        _display_data_summary(schema, summary)
        
        # Display sample data
        console.print("\n[bold green]Sample Data:[/bold green]")
        console.print(df.head())
        
        # Display data quality issues
        if summary['data_quality']['duplicate_rows'] > 0:
            console.print(f"\n[bold yellow]⚠️  Found {summary['data_quality']['duplicate_rows']} duplicate rows[/bold yellow]")
        
        if summary['data_quality']['constant_features']:
            console.print(f"\n[bold yellow]⚠️  Constant features: {summary['data_quality']['constant_features']}[/bold yellow]")
        
        if summary['data_quality']['high_missing_features']:
            console.print(f"\n[bold yellow]⚠️  High missing value features: {summary['data_quality']['high_missing_features']}[/bold yellow]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise click.Abort()


@cli.command()
def models():
    """List all available models in the zoo."""
    
    console.print(Panel.fit(
        "[bold blue]🤖 Model Zoo[/bold blue]\n"
        "Available models for classification and regression tasks",
        title="Model Registry"
    ))
    
    # Classification models
    classification_models = model_zoo.list_models(is_classification=True)
    console.print("\n[bold green]Classification Models:[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Priority", justify="center")
    table.add_column("Requires Scaling", justify="center")
    table.add_column("Handles Categorical", justify="center")
    
    for model_name in classification_models:
        config = model_zoo.get_model_config(model_name)
        table.add_row(
            model_name,
            str(config.priority),
            "✅" if config.requires_scaling else "❌",
            "✅" if config.handles_categorical else "❌"
        )
    
    console.print(table)
    
    # Regression models
    regression_models = model_zoo.list_models(is_classification=False)
    console.print("\n[bold green]Regression Models:[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Priority", justify="center")
    table.add_column("Requires Scaling", justify="center")
    table.add_column("Handles Categorical", justify="center")
    
    for model_name in regression_models:
        config = model_zoo.get_model_config(model_name)
        table.add_row(
            model_name,
            str(config.priority),
            "✅" if config.requires_scaling else "❌",
            "✅" if config.handles_categorical else "❌"
        )
    
    console.print(table)


def _display_data_summary(schema, summary):
    """Display data summary in a formatted table."""
    
    console.print("\n[bold green]📊 Dataset Summary:[/bold green]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Rows", f"{schema.n_rows:,}")
    table.add_row("Features", f"{schema.n_features}")
    table.add_row("Categorical Features", f"{schema.n_categorical}")
    table.add_row("Numerical Features", f"{schema.n_numerical}")
    table.add_row("Missing Values", f"{schema.n_missing:,}")
    table.add_row("Missing Percentage", f"{schema.missing_percentage:.2f}%")
    table.add_row("Target Type", schema.target_type)
    table.add_row("Memory Usage", f"{schema.memory_usage_mb:.2f} MB")
    
    if schema.class_balance:
        table.add_row("Class Balance", f"{len(schema.class_balance)} classes")
    
    console.print(table)


def _display_results(results, metric):
    """Display model results in a formatted table."""
    
    console.print("\n[bold green]🏆 Model Results:[/bold green]")
    
    # Sort results by best score
    results_sorted = sorted(results, key=lambda x: x['best_score'], reverse=True)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="center")
    table.add_column("Model", style="cyan")
    table.add_column("Best Score", justify="right", style="green")
    table.add_column("Training Time", justify="right")
    table.add_column("Status", justify="center")
    
    for i, result in enumerate(results_sorted, 1):
        status = "✅" if result['best_score'] > 0 else "❌"
        table.add_row(
            str(i),
            result['model'],
            f"{result['best_score']:.4f}",
            f"{result['training_time']:.2f}s",
            status
        )
    
    console.print(table)
    
    # Display best model details
    if results_sorted:
        best_result = results_sorted[0]
        console.print(f"\n[bold green]🥇 Best Model: {best_result['model']}[/bold green]")
        console.print(f"Score: {best_result['best_score']:.4f}")
        console.print(f"Parameters: {json.dumps(best_result['best_params'], indent=2)}")


def _save_results(results, output_path, schema, summary):
    """Save results to files."""
    
    # Save results JSON
    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'schema': schema.__dict__,
            'summary': summary,
            'timestamp': time.time()
        }, f, indent=2, default=str)
    
    # Save CSV summary
    results_df = pd.DataFrame([
        {
            'model': r['model'],
            'best_score': r['best_score'],
            'training_time': r['training_time'],
            'best_params': json.dumps(r['best_params'])
        }
        for r in results
    ])
    
    results_df.to_csv(output_path / "results_summary.csv", index=False)
    
    console.print(f"[dim]Results saved to {output_path}[/dim]")


if __name__ == '__main__':
    cli()
