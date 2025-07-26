# evaluate.py
import os
import re

def load_optimal_solutions():
    """Load known optimal solutions for comparison"""
    evaluation_dir = "../evaluation_dataset/extracted"
    optimal_solutions = {}
    
    for filename in os.listdir(evaluation_dir):
        if filename.endswith('.opt.tour'):
            dataset_name = filename.replace('.opt.tour', '')
            filepath = os.path.join(evaluation_dir, filename)
            
            with open(filepath, 'r') as f:
                content = f.read()
                # Extract optimal distance from comment
                match = re.search(r'\((\d+)\)', content)
                if match:
                    optimal_solutions[dataset_name] = int(match.group(1))
    
    return optimal_solutions

def calculate_error_metrics(results, dataset_names):
    """Calculate error percentages against optimal solutions"""
    optimal_solutions = load_optimal_solutions()
    
    for i, dataset_name in enumerate(dataset_names):
        if dataset_name in optimal_solutions:
            optimal_distance = optimal_solutions[dataset_name]
            print(f"\n{dataset_name} - Optimal: {optimal_distance}")
            
            for alg_name in results:
                algorithm_distance = results[alg_name][i]['distance']
                error_percentage = ((algorithm_distance - optimal_distance) / optimal_distance) * 100
                print(f"  {alg_name}: {algorithm_distance:.2f} ({error_percentage:+.2f}% from optimal)")
