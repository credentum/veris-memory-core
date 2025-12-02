#!/usr/bin/env python3
"""
Evaluation runner for semantic search quality measurement.
Produces deterministic results with seed control.
"""

import json
import csv
import hashlib
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse


class EvalRunner:
    def __init__(self, seed: int = 42):
        """Initialize with fixed seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def load_dataset(self, dataset_path: str) -> Dict:
        """Load evaluation dataset and verify integrity."""
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Verify hash if manifest exists
        manifest_path = Path(dataset_path).parent / 'manifest.json'
        if manifest_path.exists():
            with open(dataset_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            expected_hash = manifest.get('files', {}).get('eval_dataset.json', {}).get('sha256')
            if expected_hash and file_hash != expected_hash:
                print(f"WARNING: Dataset hash mismatch! Expected: {expected_hash}, Got: {file_hash}")
        
        return data
    
    def simulate_retrieval(self, query: str, documents: List[Dict], k: int = 5) -> List[str]:
        """
        Simulate retrieval by random ranking with bias toward relevant docs.
        In production, this would call the actual search API.
        """
        # Simple simulation: shuffle with bias
        doc_ids = [doc['id'] for doc in documents]
        
        # For reproducible simulation, use seeded shuffle
        random.shuffle(doc_ids)
        
        return doc_ids[:k]
    
    def calculate_precision_at_k(self, retrieved: List[str], relevant: List[str], k: int = 1) -> float:
        """Calculate Precision@K metric."""
        retrieved_k = retrieved[:k]
        relevant_in_k = len(set(retrieved_k) & set(relevant))
        return relevant_in_k / k if k > 0 else 0.0
    
    def calculate_ndcg_at_k(self, retrieved: List[str], relevance_scores: Dict[str, float], k: int = 5) -> float:
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because positions start at 1
        
        # Calculate ideal DCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate_dataset(self, dataset: Dict) -> Dict:
        """Run evaluation on a dataset."""
        results = {
            'queries': [],
            'metrics': {
                'p_at_1': [],
                'ndcg_at_5': [],
                'mrr': []
            }
        }
        
        queries = dataset.get('queries', [])
        documents = dataset.get('documents', [])
        
        for query in queries:
            # Simulate retrieval
            retrieved = self.simulate_retrieval(query['query'], documents, k=10)
            
            # Get expected results
            expected = query.get('expected_docs', [])
            relevance_scores = query.get('relevance_scores', {})
            
            # Calculate metrics
            p_at_1 = self.calculate_precision_at_k(retrieved, expected, k=1)
            ndcg_at_5 = self.calculate_ndcg_at_k(retrieved, relevance_scores, k=5)
            mrr = self.calculate_mrr(retrieved, expected)
            
            # Store results
            query_result = {
                'query_id': query['id'],
                'query': query['query'],
                'retrieved': retrieved[:5],
                'expected': expected,
                'metrics': {
                    'p_at_1': p_at_1,
                    'ndcg_at_5': ndcg_at_5,
                    'mrr': mrr
                }
            }
            
            results['queries'].append(query_result)
            results['metrics']['p_at_1'].append(p_at_1)
            results['metrics']['ndcg_at_5'].append(ndcg_at_5)
            results['metrics']['mrr'].append(mrr)
        
        # Calculate aggregate metrics
        results['aggregate'] = {
            'mean_p_at_1': np.mean(results['metrics']['p_at_1']),
            'mean_ndcg_at_5': np.mean(results['metrics']['ndcg_at_5']),
            'mean_mrr': np.mean(results['metrics']['mrr']),
            'std_p_at_1': np.std(results['metrics']['p_at_1']),
            'std_ndcg_at_5': np.std(results['metrics']['ndcg_at_5']),
            'std_mrr': np.std(results['metrics']['mrr'])
        }
        
        return results
    
    def save_results(self, results: Dict, output_path: str, format: str = 'json'):
        """Save evaluation results to file."""
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        elif format == 'csv':
            csv_path = output_path.replace('.json', '.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Mean', 'Std Dev'])
                for metric, values in results['aggregate'].items():
                    if metric.startswith('mean_'):
                        metric_name = metric.replace('mean_', '')
                        std_metric = f'std_{metric_name}'
                        writer.writerow([
                            metric_name.upper(),
                            f"{values:.4f}",
                            f"{results['aggregate'].get(std_metric, 0):.4f}"
                        ])
    
    def run(self, dataset_name: str = 'clean', output_dir: str = 'reports'):
        """Run evaluation on specified dataset."""
        # Map dataset names to paths
        dataset_paths = {
            'clean': 'eval/clean/eval_dataset.json',
            'noisy': 'eval/noisy/eval_dataset_noisy.json',
            'long': 'eval/long/eval_dataset_long.json'
        }
        
        if dataset_name == 'all':
            all_results = {}
            for name, path in dataset_paths.items():
                if Path(path).exists():
                    print(f"Evaluating {name} dataset...")
                    dataset = self.load_dataset(path)
                    results = self.evaluate_dataset(dataset)
                    all_results[name] = results
                    
                    # Save individual results
                    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                    output_path = f"{output_dir}/eval-{name}-{timestamp}.json"
                    self.save_results(results, output_path)
                    
                    # Print summary
                    print(f"  P@1: {results['aggregate']['mean_p_at_1']:.4f}")
                    print(f"  NDCG@5: {results['aggregate']['mean_ndcg_at_5']:.4f}")
                    print(f"  MRR: {results['aggregate']['mean_mrr']:.4f}")
            
            # Save comparison
            comparison_path = f"{output_dir}/eval-comparison-{timestamp}.json"
            with open(comparison_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            return all_results
        
        else:
            dataset_path = dataset_paths.get(dataset_name)
            if not dataset_path or not Path(dataset_path).exists():
                raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {dataset_path}")
            
            print(f"Evaluating {dataset_name} dataset...")
            dataset = self.load_dataset(dataset_path)
            results = self.evaluate_dataset(dataset)
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            output_path = f"{output_dir}/eval-{dataset_name}-{timestamp}.json"
            Path(output_dir).mkdir(exist_ok=True)
            self.save_results(results, output_path)
            self.save_results(results, output_path, format='csv')
            
            # Print summary
            print(f"\nResults for {dataset_name} dataset:")
            print(f"  P@1: {results['aggregate']['mean_p_at_1']:.4f} (±{results['aggregate']['std_p_at_1']:.4f})")
            print(f"  NDCG@5: {results['aggregate']['mean_ndcg_at_5']:.4f} (±{results['aggregate']['std_ndcg_at_5']:.4f})")
            print(f"  MRR: {results['aggregate']['mean_mrr']:.4f} (±{results['aggregate']['std_mrr']:.4f})")
            print(f"\nResults saved to: {output_path}")
            
            return results


def main():
    parser = argparse.ArgumentParser(description='Run evaluation on semantic search datasets')
    parser.add_argument('--dataset', default='clean', choices=['clean', 'noisy', 'long', 'all'],
                        help='Dataset to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', default='reports',
                        help='Output directory for results')
    parser.add_argument('--compare', action='store_true',
                        help='Compare results across datasets')
    
    args = parser.parse_args()
    
    runner = EvalRunner(seed=args.seed)
    results = runner.run(dataset_name=args.dataset, output_dir=args.output)
    
    if args.compare and args.dataset == 'all':
        print("\n" + "="*50)
        print("COMPARISON ACROSS DATASETS")
        print("="*50)
        for dataset_name, dataset_results in results.items():
            print(f"\n{dataset_name.upper()}:")
            for metric, value in dataset_results['aggregate'].items():
                if metric.startswith('mean_'):
                    print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()