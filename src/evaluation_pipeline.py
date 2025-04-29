# Standard library imports
import argparse
import json
import time
from datetime import datetime

# Related third-party imports
import matplotlib.pyplot as plt
import pandas as pd

# Local application/library specific imports
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_evaluator import ModelEvaluator

def run_evaluation(language=None, model_name="all-MiniLM-L6-v2", small_version=True, 
                  split_column='split', batch_size=128, save_results=True,
                  example_path="../dataset/shopping_queries_dataset_examples.parquet",
                  product_path="../dataset/shopping_queries_dataset_products.parquet",
                  product_size=None):
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
        small_version=small_version
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

def main():
    """
    Main function to run the evaluation pipeline
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run model evaluation on shopping queries dataset')
    parser.add_argument('--model', type=str, default="all-MiniLM-L6-v2",
                      help='Model name to evaluate (default: all-MiniLM-L6-v2)')
    parser.add_argument('--language', type=str, default=None,
                      help='Language to filter data by (default: None for all languages)')
    parser.add_argument('--dataset_size', choices=['small', 'large'], default='small',
                   help='Size of dataset to use (default: small)')
    parser.add_argument('--split_column', type=str, default='split',
                     help='Column name for train/test split (default: "split")')
    parser.add_argument('--batch_size', type=int, default=128,
                     help='Batch size for evaluation (default: 128)')
    parser.add_argument('--example_path', type=str, default="../dataset/shopping_queries_dataset_examples.parquet",
                     help='Path to the examples dataset (default: ../dataset/shopping_queries_dataset_examples.parquet)')
    parser.add_argument('--product_path', type=str, default="../dataset/shopping_queries_dataset_products.parquet",
                     help='Path to the products dataset (default: ../dataset/shopping_queries_dataset_products.parquet)')
    parser.add_argument('--product_size', type=int, default=None,
                     help='Size of the product dataset to use (default: None for only the query products itself without any additional products)')
    
    args = parser.parse_args()
    
    # Run evaluation based on arguments
    print(f"Running single model evaluation for model: {args.model}")
    print(f"Language: {args.language if args.language else 'all'}")
    
    results = run_evaluation(
        language=args.language,
        model_name=args.model,
        small_version=True if args.dataset_size == "small" else False,
        split_column=args.split_column,
        batch_size=args.batch_size,
        save_results=True,
        example_path=args.example_path,
        product_path=args.product_path,
        product_size=args.product_size
    )
    
    # Print the results
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()