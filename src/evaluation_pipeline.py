# Standard library imports
import argparse
import json
import time
from datetime import datetime
import os
from typing import List, Dict, Any

# Related third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Local application/library specific imports
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_evaluator import ModelEvaluator

def run_evaluation(language=None, model_name="all-MiniLM-L6-v2", small_version=True, 
                  split_column='split', batch_size=128, save_results=True,
                  example_path="../dataset/shopping_queries_dataset_examples.parquet",
                  product_path="../dataset/shopping_queries_dataset_products.parquet",
                  product_size=None,dataset_size_ratio=1.0):
    """
    Run the complete evaluation pipeline with comprehensive metrics tracking
    """
    # Start timing the evaluation
    start_time = time.time()
    
    # Initialize results dictionary
    evaluation_results = {
        "model_name": model_name,
        "language": language if language else "all",
        "dataset_size": "small" if small_version else "full",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_stats": {},
        "train_metrics": {},
        "test_metrics": {},
        "performance_metrics": {}
    }
    print(f"Running on {'small' if small_version else 'full'} version of the dataset")
    # Initialize data preprocessor
    if language is None:
        print(f"Initializing data preprocessor for all languages")
    else:
        print(f"Initializing data preprocessor for language: {language}")
    
    # Load data
    data_load_start = time.time()
    loaded_data = DataLoader(
        example_path=example_path,
        products_path=product_path,
        small_version=small_version,
        dataset_size_ratio=dataset_size_ratio
    )
    loaded_data.load_data()
    data_load_time = time.time() - data_load_start
    evaluation_results["performance_metrics"]["data_loading_time"] = data_load_time

    # Preprocess data
    preprocess_start = time.time()
    preprocessor = DataPreprocessor(language=language, df=loaded_data.df)
    preprocessor.data_preprocessing()
    
    # Filter by locale
    if (language is not None):
        preprocessor.filter_by_locale()
    
    # Print the first few rows of the preprocessed data
    print(preprocessor.df.head())
    # Get dataset statistics
    stats = preprocessor.get_dataset_stats()
    evaluation_results["dataset_stats"] = stats
    
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Unique queries: {stats['unique_queries']}")
    print(f"Average products per query: {stats['avg_products_per_query']:.2f}")
    print("Relevance distribution:")
    for score, pct in stats['relevance_distribution'].items():
        print(f"  Score {score}: {pct:.2%}")
    
    # Split data
    train_df, test_df = preprocessor.split_data(split_column)
    preprocess_time = time.time() - preprocess_start
    evaluation_results["performance_metrics"]["preprocessing_time"] = preprocess_time
    product_count = product_size if product_size is not None else ""
    # Initialize model evaluator
    print(f"\nInitializing model evaluator with model: {model_name}")
    model_init_start = time.time()
    evaluator = ModelEvaluator(model_name=model_name,batch_size=batch_size)
    model_init_time = time.time() - model_init_start
    evaluation_results["performance_metrics"]["model_initialization_time"] = model_init_time
    
    # Evaluate train model
    print(f"\nEvaluating model on {language if language else 'all languages'} train set...")
    print(train_df.head())
    train_eval_start = time.time()
    train_results = evaluator.evaluate(train_df, threshold=3.0,product_size=product_size)  # Consider 'E' and 'S' as relevant (4 and 3)
    train_eval_time = time.time() - train_eval_start
    evaluation_results["performance_metrics"]["train_evaluation_time"] = train_eval_time
    evaluation_results["train_metrics"] = train_results
    
    evaluator.print_results(train_results, language=language, small_version=small_version)

    # Evaluate test model
    print(f"Evaluating model on {language if language else 'all languages'} test set...")
    print(test_df.head())
    test_eval_start = time.time()
    test_results = evaluator.evaluate(test_df=test_df, threshold=3.0,product_size=product_size)  # Consider 'E' and 'S' as relevant (4 and 3)
    test_eval_time = time.time() - test_eval_start
    evaluation_results["performance_metrics"]["test_evaluation_time"] = test_eval_time
    evaluation_results["test_metrics"] = test_results
    
    evaluator.print_results(test_results, language=language, small_version=small_version)
    
    
    # Calculate total time
    total_time = time.time() - start_time
    evaluation_results["performance_metrics"]["total_evaluation_time"] = total_time
    
    print(f"\nTotal evaluation time: {total_time:.2f} seconds")
    
    # Generate visualization of metrics
    if save_results:
        # Create bar chart comparing train and test metrics
        metrics = ['NDCG@10', 'Recall@10', 'MRR@10']
        train_values = [train_results[metric] for metric in metrics]
        test_values = [test_results[metric] for metric in metrics]
        
        x = range(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        train_bars = ax.bar([i - width/2 for i in x], train_values, width, label='Train')
        test_bars = ax.bar([i + width/2 for i in x], test_values, width, label='Test')
        
        # Add values on top of bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
        
        add_labels(train_bars)
        add_labels(test_bars)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, max(max(train_values), max(test_values)) * 1.1)
        ax.set_title(f'Model Performance: {model_name}')
        ax.legend()
        
        # Save the visualization
        fig_filename = f"output/{model_name}_{language if language else 'all'}_{'small' if small_version else 'large'}_{product_count}_metrics.png"
        plt.savefig(fig_filename)
        evaluation_results["visualization_file"] = fig_filename
        
        # Save results to JSON file
        json_filename = f"output/{model_name}_{language if language else 'all'}_{'small' if small_version else 'large'}_{product_count}_results.json"
        with open(json_filename, 'w') as f:
            # Convert sets to lists for JSON serialization if needed
            results_json = {k: (list(v) if isinstance(v, set) else v) 
                           for k, v in evaluation_results.items()}
            json.dump(results_json, f, indent=4)
        
        # Save results to CSV for easy import to spreadsheets
        csv_data = {
            'Model': model_name,
            'Language': language if language else 'all',
            'Dataset Size': 'Small' if small_version else 'Full',
            'Total Samples': stats['total_samples'],
            'Unique Queries': stats['unique_queries'],
            'Train NDCG@10': train_results['NDCG@10'],
            'Train Recall@10': train_results['Recall@10'],
            'Train MRR@10': train_results['MRR@10'],
            'Test NDCG@10': test_results['NDCG@10'],
            'Test Recall@10': test_results['Recall@10'],
            'Test MRR@10': test_results['MRR@10'],
            'Total Time (s)': total_time
        }
        
        pd.DataFrame([csv_data]).to_csv(f"output/{model_name}_{language if language else 'all'}_{'small' if small_version else 'large'}_{product_count}_results.csv", index=False)
        
        print(f"Results saved to {json_filename} and {fig_filename}")
    
    print("\n")
    
    return evaluation_results
def run_multi_model_evaluation(models: List[str], **kwargs) -> Dict[str, Any]:
    """
    Run evaluation for multiple models and generate analysis
    """
    print(f"Starting multi-model evaluation for {len(models)} models")
    print(f"Models: {', '.join(models)}")
    print("=" * 80)
    
    all_results = {}
    comparative_data = []
    total_start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Run evaluation for each model
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Evaluating model: {model}")
        print("-" * 60)
        
        try:
            model_results = run_evaluation(model_name=model, **kwargs)
            all_results[model] = model_results
            
            # Collect data for comparison
            comparative_row = {
                'Model': model,
                'Language': model_results['language'],
                'Dataset_Size': model_results['dataset_size'],
                'Total_Samples': model_results['dataset_stats']['total_samples'],
                'Train_NDCG@10': model_results['train_metrics']['NDCG@10'],
                'Train_Recall@10': model_results['train_metrics']['Recall@10'],
                'Train_MRR@10': model_results['train_metrics']['MRR@10'],
                'Test_NDCG@10': model_results['test_metrics']['NDCG@10'],
                'Test_Recall@10': model_results['test_metrics']['Recall@10'],
                'Test_MRR@10': model_results['test_metrics']['MRR@10'],
                'Total_Time': model_results['performance_metrics']['total_evaluation_time'],
                'Model_Init_Time': model_results['performance_metrics']['model_initialization_time']
            }
            comparative_data.append(comparative_row)
            
            print(f"Successfully evaluated {model}")
            
        except Exception as e:
            print(f"Error evaluating {model}: {str(e)}")
            all_results[model] = {"error": str(e)}
            continue
    
    total_time = time.time() - total_start_time
    
    # Create comparative analysis
    if comparative_data:
        comparison_df = pd.DataFrame(comparative_data)
        
        # Generate comparative visualizations
        generate_comparative_visualizations(comparison_df, kwargs)
        
        # Save comparative results
        save_comparative_results(comparison_df, all_results, total_time, kwargs)
        
        # Print summary
        print_evaluation_summary(comparison_df, total_time)
    
    return {
        'individual_results': all_results,
        'comparative_data': comparative_data,
        'total_evaluation_time': total_time,
        'successful_models': len(comparative_data),
        'failed_models': len(models) - len(comparative_data)
    }

def generate_comparative_visualizations(df: pd.DataFrame, config: Dict[str, Any]):
    """Generate comparative visualizations for multiple models"""
    
    language = config.get('language', 'all')
    dataset_size = 'small' if config.get('small_version', True) else 'large'
    product_size = config.get('product_size', '')
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Multi-Model Performance Comparison\nLanguage: {language}, Dataset: {dataset_size}', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['Test_NDCG@10', 'Test_Recall@10', 'Test_MRR@10']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Bar chart for test metrics
    ax1 = axes[0, 0]
    x = range(len(df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        bars = ax1.bar([j + offset for j in x], df[metric], width, 
                      label=metric.replace('Test_', ''), color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Test Set Performance Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training vs Test Performance
    ax2 = axes[0, 1]
    train_ndcg = df['Train_NDCG@10']
    test_ndcg = df['Test_NDCG@10']
    
    ax2.scatter(train_ndcg, test_ndcg, s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
    
    # Add diagonal line for reference
    min_val = min(train_ndcg.min(), test_ndcg.min())
    max_val = max(train_ndcg.max(), test_ndcg.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    # Add model labels
    for i, model in enumerate(df['Model']):
        ax2.annotate(model, (train_ndcg.iloc[i], test_ndcg.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Train NDCG@10')
    ax2.set_ylabel('Test NDCG@10')
    ax2.set_title('Train vs Test Performance (NDCG@10)')
    ax2.grid(True, alpha=0.3)
    
    # Execution Time Comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(df['Model'], df['Total_Time'], color='skyblue', alpha=0.7)
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Total Evaluation Time')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                f'{height:.1f}s', ha='center', va='bottom')
    
    # Performance vs Time Trade-off
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['Total_Time'], df['Test_NDCG@10'], 
                         s=100, alpha=0.7, c=range(len(df)), cmap='plasma')
    
    for i, model in enumerate(df['Model']):
        ax4.annotate(model, (df['Total_Time'].iloc[i], df['Test_NDCG@10'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Total Time (seconds)')
    ax4.set_ylabel('Test NDCG@10')
    ax4.set_title('Performance vs Time Trade-off')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparison plot
    filename = f"output/multi_model_comparison_{language}_{dataset_size}_{product_size}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed metrics heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select metrics for heatmap
    heatmap_columns = ['Train_NDCG@10', 'Train_Recall@10', 'Train_MRR@10',
                      'Test_NDCG@10', 'Test_Recall@10', 'Test_MRR@10']
    heatmap_data = df[heatmap_columns].set_index(df['Model'])
    
    # Create heatmap
    sns.heatmap(heatmap_data.T, annot=True, cmap='YlOrRd', fmt='.4f', 
                cbar_kws={'label': 'Score'}, ax=ax)
    ax.set_title(f'Model Performance Heatmap\nLanguage: {language}, Dataset: {dataset_size}')
    ax.set_ylabel('Metrics')
    ax.set_xlabel('Models')
    
    plt.tight_layout()
    heatmap_filename = f"output/multi_model_heatmap_{language}_{dataset_size}_{product_size}.png"
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparative visualizations saved:")
    print(f"  - {filename}")
    print(f"  - {heatmap_filename}")

def save_comparative_results(df: pd.DataFrame, all_results: Dict, total_time: float, config: Dict):
    """Save comparative results to files"""
    
    language = config.get('language', 'all')
    dataset_size = 'small' if config.get('small_version', True) else 'large'
    product_size = config.get('product_size', '')
    
    # Save comparative CSV
    csv_filename = f"output/multi_model_comparison_{language}_{dataset_size}_{product_size}.csv"
    df.to_csv(csv_filename, index=False)
    
    # Save detailed JSON results
    json_filename = f"output/multi_model_detailed_results_{language}_{dataset_size}_{product_size}.json"
    detailed_results = {
        'evaluation_config': config,
        'total_evaluation_time': total_time,
        'models_evaluated': len(df),
        'comparative_summary': df.to_dict('records'),
        'detailed_results': all_results,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(json_filename, 'w') as f:
        json.dump(detailed_results, f, indent=4, default=str)
    
    # Create ranking summary
    ranking_data = []
    for metric in ['Test_NDCG@10', 'Test_Recall@10', 'Test_MRR@10']:
        sorted_df = df.sort_values(metric, ascending=False)
        for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
            ranking_data.append({
                'Metric': metric,
                'Rank': rank,
                'Model': row['Model'],
                'Score': row[metric]
            })
    
    ranking_df = pd.DataFrame(ranking_data)
    ranking_filename = f"output/multi_model_rankings_{language}_{dataset_size}_{product_size}.csv"
    ranking_df.to_csv(ranking_filename, index=False)
    
    print(f"Comparative results saved:")
    print(f"  - Summary: {csv_filename}")
    print(f"  - Detailed: {json_filename}")
    print(f"  - Rankings: {ranking_filename}")

def print_evaluation_summary(df: pd.DataFrame, total_time: float):
    """Print the summary of the evaluation results"""
    
    print("\n" + "="*80)
    print("MULTI-MODEL EVALUATION SUMMARY")
    print("="*80)
    
    print(f"Total Evaluation Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Models Evaluated: {len(df)}")
    print(f"Average Time per Model: {total_time/len(df):.2f} seconds")
    
    print("\n" + "-"*60)
    print("TOP PERFORMERS BY METRIC:")
    print("-"*60)
    
    metrics = ['Test_NDCG@10', 'Test_Recall@10', 'Test_MRR@10']
    for metric in metrics:
        best_model = df.loc[df[metric].idxmax()]
        print(f"{metric:15} | {best_model['Model']:25} | {best_model[metric]:.4f}")
    
    print("\n" + "-"*60)
    print("EFFICIENCY ANALYSIS:")
    print("-"*60)
    fastest_model = df.loc[df['Total_Time'].idxmin()]
    print(f"Fastest Model: {fastest_model['Model']} ({fastest_model['Total_Time']:.2f}s)")
    
    # Calculate efficiency score (NDCG/Time)
    df['Efficiency'] = df['Test_NDCG@10'] / df['Total_Time']
    most_efficient = df.loc[df['Efficiency'].idxmax()]
    print(f"Most Efficient: {most_efficient['Model']} (Score: {most_efficient['Efficiency']:.6f})")
    
    print("\n" + "-"*60)
    print("PERFORMANCE STATISTICS:")
    print("-"*60)
    for metric in metrics:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        print(f"{metric:15} | Mean: {mean_val:.4f} | Std: {std_val:.4f}")

def main():
    """
    Main function to run single or multi-model evaluation
    """
    parser = argparse.ArgumentParser(description='Run model evaluation on shopping queries dataset')
    parser.add_argument('--models', type=str, nargs='+', default=["all-MiniLM-L6-v2"],
                      help='Model names to evaluate (space-separated for multiple models)')
    parser.add_argument('--model', type=str, default=None,
                      help='Single model name (deprecated, use --models instead)')
    parser.add_argument('--language', type=str, default=None,
                      help='Language to filter data by (default: None for all languages)')
    parser.add_argument('--dataset_size', choices=['small', 'large'], default='small',
                      help='Size of dataset to use (default: small)')
    parser.add_argument('--split_column', type=str, default='split',
                      help='Column name for train/test split (default: "split")')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for evaluation (default: 128)')
    parser.add_argument('--example_path', type=str, default="../dataset/shopping_queries_dataset_examples.parquet",
                      help='Path to the examples dataset')
    parser.add_argument('--product_path', type=str, default="../dataset/shopping_queries_dataset_products.parquet",
                      help='Path to the products dataset')
    parser.add_argument('--product_size', type=int, default=None,
                      help='Size of the product dataset to use (default: None)')
    parser.add_argument('--dataset_size_ratio', type=float, default=1.0,
                      help='Size of the dataset wanted to train and test')
    
    args = parser.parse_args()
    
    # Handle backward compatibility
    if args.model and not args.models:
        models = [args.model]
    else:
        models = args.models
    
    # Prepare evaluation parameters
    eval_params = {
        'language': args.language,
        'small_version': args.dataset_size == 'small',
        'split_column': args.split_column,
        'batch_size': args.batch_size,
        'save_results': True,
        'example_path': args.example_path,
        'product_path': args.product_path,
        'product_size': args.product_size,
        'dataset_size_ratio': args.dataset_size_ratio
    }
    
    # Run evaluation
    if len(models) == 1:
        print(f"Running single model evaluation for: {models[0]}")
        results = run_evaluation(model_name=models[0], **eval_params)
        print(json.dumps(results, indent=4, default=str))
    else:
        print(f"Running multi-model evaluation for {len(models)} models")
        results = run_multi_model_evaluation(models, **eval_params)
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved in output/ directory")

if __name__ == "__main__":
    main()