import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics import ndcg_score

class ModelEvaluator:
    def __init__(self, model_name="all-MiniLM-L6-v2", use_gpu=True, batch_size=128):
        """
        Initialize the model evaluator
        """
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        print(f"Using device: {self.device}")
        print(f"Loaded model: {model_name}")
        
    @staticmethod
    def calculate_ndcg(relevance_scores, k=10):
        """
        Calculate NDCG@k for a single query
        """
        if len(relevance_scores) == 0:
            return 0.0
        
        # Limit to top k scores
        relevance_scores = relevance_scores[:k]
        
        # Calculate DCG
        dcg = 0
        for i, score in enumerate(relevance_scores):
            dcg += (2 ** score - 1) / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate ideal DCG
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0
        for i, score in enumerate(ideal_scores):
            idcg += (2 ** score - 1) / np.log2(i + 2)
        
        # Avoid division by zero
        if idcg == 0:
            return 0.0
        
        # Return NDCG
        return dcg / idcg
    
    @staticmethod
    def calculate_recall_at_k(relevant_items, retrieved_items, k=10):
        """
        Calculate Recall@k for a single query
        """
        if len(relevant_items) == 0:
            return 0.0
        
        # Limit to top k retrieved items
        retrieved_items_at_k = set(retrieved_items[:k])
        
        # Calculate recall
        return len(retrieved_items_at_k.intersection(relevant_items)) / len(relevant_items)
    
    @staticmethod
    def calculate_mrr(retrieved_items, relevant_items, k=10):
        """
        Calculate MRR@k for a single query
        """
        # Limit to top k retrieved items
        retrieved_items = retrieved_items[:k]
        
        # Find the first relevant item
        for i, item in enumerate(retrieved_items, start=1):
            if item in relevant_items:
                return 1.0 / (i)
        
        return 0.0
    
    def evaluate(self, test_df, threshold=3.0, k=10):
        """
        Evaluate the model on the test set
        """
        # Group the test data by query
        query_groups = test_df.groupby('query')

        # Adjust batch size based on GPU memory
        if self.device.type == 'cuda':
            # Smaller batch size for GPU to avoid out-of-memory errors
            print("Running on GPU...")
            if self.batch_size > 128:
                print(f"Reducing batch size from {self.batch_size} to 128 for GPU processing")
                self.batch_size = 128
        
        # Track metrics for each query
        ndcg_scores = []
        recall_scores = []
        mrr_scores = []
        
        # Process queries in batches
        unique_queries = list(query_groups.groups.keys())
        
        for i in tqdm(range(0, len(unique_queries), self.batch_size), desc="Evaluating queries"):
            batch_queries = unique_queries[i:i+self.batch_size]
            
            # Collect all products for the batch queries
            batch_data = []
            for query in batch_queries:
                query_df = query_groups.get_group(query)
                query_products = query_df['combined_text'].tolist()
                query_relevances = query_df['relevance'].tolist()
                batch_data.append((query, query_products, query_relevances))
            
            # Encode all unique queries in the batch
            query_texts = [item[0] for item in batch_data]

            query_embeddings = self.model.encode(
                query_texts, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                device=self.device
            )
            
            # Process each query in the batch
            for idx, (query, products, relevances) in enumerate(batch_data):
                # Skip if no products
                if len(products) == 0:
                    continue
                
                # Encode products
                product_embeddings = self.model.encode(products, convert_to_tensor=True, show_progress_bar=False
                ,device=self.device)
                
                # Calculate similarities
                q_embedding = query_embeddings[idx].unsqueeze(0)  # Add batch dimension
                similarities = util.pytorch_cos_sim(q_embedding, product_embeddings)[0].cpu().numpy()
                
                # Sort products by similarity
                sorted_indices = np.argsort(-similarities)
                
                # Get sorted relevance scores
                sorted_relevances = [relevances[idx] for idx in sorted_indices]
                
                # Calculate NDCG@k
                ndcg = self.calculate_ndcg(sorted_relevances, k)
                ndcg_scores.append(ndcg)
                
                # Calculate Recall@k and MRR@k
                relevant_indices = {i for i, rel in enumerate(relevances) if rel >= threshold}
                retrieved_indices = sorted_indices.tolist()
                
                recall = self.calculate_recall_at_k(relevant_indices, retrieved_indices, k)
                recall_scores.append(recall)
                
                mrr = self.calculate_mrr(retrieved_indices, relevant_indices, k)
                mrr_scores.append(mrr)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Calculate average metrics
        avg_ndcg = np.mean(ndcg_scores)
        avg_recall = np.mean(recall_scores)
        avg_mrr = np.mean(mrr_scores)
        
        # Return metrics
        return {
            'NDCG@10': avg_ndcg,
            'Recall@10': avg_recall,
            'MRR@10': avg_mrr,
            'Number of Queries': len(unique_queries)
        }
    def print_results(self,results,language=None,small_version=None):
        """
        Print evaluation results
        """
        print("\nEvaluation Results:")
        print("-" * 50)
        print(f"Model: {self.model_name}")
        print(f"Dataset: ESCI {language.upper() if language else 'All Languages'} {'Small' if small_version else 'Large'}")
        print(f"Number of test queries: {results['Number of Queries']}")
        print("-" * 50)
        print(f"NDCG@10: {results['NDCG@10']:.4f}")
        print(f"Recall@10: {results['Recall@10']:.4f}")
        print(f"MRR@10: {results['MRR@10']:.4f}")