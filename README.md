# Evaluating a Pre-trained Model for Semantic Search (Retrieval)

## Introduction

This project evaluates pre-trained language models for semantic search in an e-commerce setting without any fine-tuning. The system supports both single-model evaluation and multi-model comparison to identify the best performing model for retrieval tasks.

The default model, [**all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from the Sentence-Transformers library, was chosen for its efficiency and strong performance in retrieval tasks. However, the framework allows for comprehensive evaluation and comparison of multiple sentence transformer models, including:

- **all-MiniLM-L6-v2**: Lightweight and efficient for fast inference
- **all-mpnet-base-v2**: Higher accuracy with moderate computational cost
- **paraphrase-MiniLM-L6-v2**: Optimized for paraphrase detection and semantic similarity
- **distiluse-base-multilingual-cased**: Multilingual support for cross-language retrieval

The multi-model evaluation feature enables comparison of different models' performance on the same dataset, providing insights into trade-offs between accuracy, speed, and resource consumption across various sentence transformer architectures.

The [**Shopping Queries Data Set (ESCI)**](https://github.com/amazon-science/esci-data) was used, preprocessed into a [query, title, description, relevance] format, and split into train and test sets.

The model was evaluated using **NDCG@10**, **Recall@10**, and **MRR@10** metrics.  
Performance was analyzed across different **languages** (e.g., English), **dataset sizes** (small vs. full) and **product sizes per query** (query's own product size or 100 (query's own product + randomly selected accross all dataset)) to better understand the model’s behavior under different data conditions.

All evaluation code and results are documented in **report_notebook.ipynb**, and the evaluation outputs are saved in the **output** folder.

## Code Structure

The project is organized with the following structure:
```bash
Semantic-Search-Evaluation/
├── dataset/                     # Contains ESCI dataset
├── src/
│   ├── data_loader.py            # Handles loading the dataset
│   ├── data_preprocessor.py      # Preprocesses the raw data for model input
│   ├── model_evaluator.py        # Contains metrics and evaluation functions
│   └── evaluation_pipeline.py    # Main function and evaluation pipeline coordinating the evaluation process
├── notebooks/
│   └── report_notebook.ipynb     # Report notebook to display results
├── output/                       # Folder for model evaluation result storage
├── requirements.txt              # Requirements file
└── README.md                     # Instructions on how to run the code
```

## Pulling the Code
To get started with the project, first clone the repository from GitHub:
> **Note:**  
> This might take a bit longer since the dataset is large (over 1GB)

```bash
git clone https://github.com/wei-wang11/Semantic-Search-Evaluation.git
cd Semantic-Search-Evaluation
```

## Running the code
To set up and run the project, follow these steps:

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

- On **Windows**:

```bash
venv\Scripts\activate
```

- On **macOS/Linux**:

```bash
source venv/bin/activate
```

### 3. Install Required Packages

Install all necessary dependencies from the `requirements.txt` file:

> **Note:**  
> You may need to adjust the `torch`, `torchaudio`, and `torchvision` versions to match your local CUDA version.  
> For example, for CUDA 12.8, use:
> ```
> torch==2.7.0+cu128
> ```
>  
> If you are not using a GPU, you can install the CPU-only versions of these packages instead.

or just run 
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Install other requirements with:

```bash
pip install -r requirements.txt
```
### 4. Run the Evaluation Pipeline

Execute the evaluation pipeline with the following command:

```bash
# Example of running the model with US locale with small dataset
python src\evaluation_pipeline.py --model all-MiniLM-L6-v2 --language us --dataset_size small --batch_size 128 --example_path dataset/shopping_queries_dataset_examples.parquet --product_path dataset/shopping_queries_dataset_products.parquet
```

```bash
# Example of running the model with JP locale with small dataset with 100 products per query
python src\evaluation_pipeline.py --model all-MiniLM-L6-v2 --language jp --dataset_size small --batch_size 2048 --example_path dataset/shopping_queries_dataset_examples.parquet --product_path dataset/shopping_queries_dataset_products.parquet --product_size 100
```

```bash
# Example of running the model with JP locale with small dataset
python src\evaluation_pipeline.py --model all-MiniLM-L6-v2 --language jp --dataset_size small --batch_size 128 --example_path dataset/shopping_queries_dataset_examples.parquet --product_path dataset/shopping_queries_dataset_products.parquet
```

```bash
# Example of running the model with all locale with large dataset
python src\evaluation_pipeline.py --model all-MiniLM-L6-v2 --dataset_size large --batch_size 128 --example_path dataset/shopping_queries_dataset_examples.parquet --product_path dataset/shopping_queries_dataset_products.parquet
```

```bash
# Example of runing a multi-model evaluation with 10% of small version dataset for all languages with 100 products per query
python src\evaluation_pipeline.py --models "all-MiniLM-L6-v2" "paraphrase-MiniLM-L6-v2" --dataset_size small --batch_size 1024 --dataset_size_ratio 0.1 --product_size 100 --example_path dataset/shopping_queries_dataset_examples.parquet --product_path dataset/shopping_queries_dataset_products.parquet
```
- `--models`: Names of multiple pre-trained models (from Sentence-Transformers). (e.g. --models "all-MiniLM-L6-v2" "paraphrase-MiniLM-L6-v2")
- `--model`: Name of the pre-trained model (from Sentence-Transformers).
- `--language`: Target language/locale (`us`, `jp`, `es`).
- `--dataset_size`: Dataset version to use (`small` or `large`).
- `--batch_size`: Batch size for encoding during evaluation.
- `--example_path`: File path for example dataset.
- `--product_path`: File path for product dataset.
- `--product_size `: Product size used to evaluate each query.

All evaluation results and outputs will be saved automatically in the `output/` folder.

### 5. Run the Notebook

Open and run the `report_notebook.ipynb` notebook to see the results and evaluations.  
All output files (metrics, evaluation results) are saved in the `output` folder.

At the end of the notebook, I have also provided the code to run the evluation. 
> **Note:**  
> Plese make sure you are running the notebook in venv

## Running the Model on GPU with PyTorch

This guide provides instructions for running a PyTorch model on a GPU.

### Prerequisites

1. Install PyTorch with GPU support. Follow the [official installation guide](https://pytorch.org/get-started/locally/).
2. Ensure you have a compatible GPU and the necessary drivers (e.g., NVIDIA CUDA Toolkit).

### Steps to Run the Model on GPU

1. **Check GPU Availability**  
    Verify that PyTorch detects your GPU:
    ```python
    import torch
    print(torch.cuda.is_available())  # Should return True if GPU is available
    ```
2. **GPU is set to true by default, if not find then will set to cpu**  
```python
import torch

# GPU is set to True by default
use_gpu = True

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")
```

### Notes

- Always check GPU memory usage to avoid out-of-memory errors.
- Use `torch.cuda.empty_cache()` to clear unused memory if needed.

For more details, refer to the [PyTorch documentation](https://pytorch.org/docs/).  