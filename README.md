# Evaluating a Pre-trained Model for Semantic Search (Retrieval)

## Introduction

This project evaluates a pre-trained language model for semantic search in an e-commerce setting without any fine-tuning.  
The selected model, [**all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from the Sentence-Transformers library, was chosen for its efficiency and strong performance in retrieval tasks.

The [**Shopping Queries Data Set (ESCI)**](https://github.com/amazon-science/esci-data) was used, preprocessed into a [query, title, description, relevance] format, and split into train and test sets.

The model was evaluated using **NDCG@10**, **Recall@10**, and **MRR@10** metrics.  
Performance was analyzed across different **languages** (e.g., English) and **dataset sizes** (small vs. full) to better understand the model’s behavior under different data conditions.

All evaluation code and results are documented in **report_notebook.ipynb**, and the evaluation outputs are saved in the **output** folder.

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
> torchaudio==2.7.0+cu128
> torchvision==0.22.0+cu128
> ```
>  
> If you are not using a GPU, you can install the CPU-only versions of these packages instead.

or just run 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install other requirements with:

```bash
pip install -r requirements.txt
```
### 4. Run the Evaluation Pipeline

Execute the evaluation pipeline with the following command:

```bash
# Go to the code file
cd src
# Example of running the model with US locale with full dataset
python evaluation_pipeline.py --model all-MiniLM-L6-v2 --language us --dataset_size large --batch_size 128
```

```bash
# Go to the code file
cd src
# Example of running the model with all locale and small dataset
python evaluation_pipeline.py --model all-MiniLM-L6-v2 --dataset_size small --batch_size 128
```

- `--model`: Name of the pre-trained model (from Sentence-Transformers).
- `--language`: Target language/locale (`us`, `jp`, `es`).
- `--dataset_size`: Dataset version to use (`small` or `large`).
- `--batch_size`: Batch size for encoding during evaluation.

All evaluation results and outputs will be saved automatically in the `output/` folder.

### 5. Run the Notebook

Open and run the `report_notebook.ipynb` notebook to see the results and evaluations.  
All output files (metrics, evaluation results) are saved in the `output` folder.

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