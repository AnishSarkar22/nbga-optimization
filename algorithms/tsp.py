# MAIN ALGORITHM for TSP_Comparison.py

# Neighborhood-Based Genetic Algorithm (NBGA) comparison with SWAP_GATSP, OX_SIM, and MOC_SIM 
# applied to the Traveling Salesman Problem (TSP)


import numpy as np
import matplotlib.pyplot as plt
import gzip
import random
import math
import time
from pathlib import Path
import os

class TSPParser:
    """Parser for TSPLIB format files, handles both .tsp and .gz files"""
    
    @staticmethod
    def parse_tsp_file(filepath):
        """Parse TSP file and return distance matrix and coordinates"""
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt') as f:
                content = f.read()
        else:
            with open(filepath, 'r') as f:
                content = f.read()
        
        lines = content.strip().split('\n')
        
        # Parse header information
        dimension = 0
        edge_weight_type = None
        edge_weight_format = None
        
        for line in lines:
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                edge_weight_type = line.split(':')[1].strip()
            elif line.startswith('EDGE_WEIGHT_FORMAT'):
                edge_weight_format = line.split(':')[1].strip()
        
        # Parse data based on format
        if edge_weight_type == 'EUC_2D':
            return TSPParser._parse_euc_2d(lines, dimension)
        elif edge_weight_type == 'EXPLICIT':
            return TSPParser._parse_explicit(lines, dimension, edge_weight_format)
        else:
            raise ValueError(f"Unsupported edge weight type: {edge_weight_type}")
    
    @staticmethod
    def _parse_euc_2d(lines, dimension):
        """Parse EUC_2D format (coordinates given)"""
        coords = []
        reading_coords = False
        
        for line in lines:
            if line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
                continue
            elif line.startswith('EOF') or line.startswith('DISPLAY_DATA_SECTION'):
                break
            elif reading_coords:
                parts = line.strip().split()
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])
        
        coords = np.array(coords)
        # Calculate Euclidean distance matrix
        dist_matrix = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                if i != j:
                    dist_matrix[i][j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
        
        return dist_matrix, coords
    
    @staticmethod
    def _parse_explicit(lines, dimension, edge_weight_format):
        """Parse explicit distance matrix"""
        distances = []
        reading_weights = False
        
        for line in lines:
            if line.startswith('EDGE_WEIGHT_SECTION'):
                reading_weights = True
                continue
            elif line.startswith('DISPLAY_DATA_SECTION') or line.startswith('EOF'):
                break
            elif reading_weights:
                distances.extend([int(x) for x in line.strip().split()])
        
        # Build distance matrix based on format
        dist_matrix = np.zeros((dimension, dimension))
        
        if edge_weight_format == 'UPPER_ROW':
            idx = 0
            for i in range(dimension):
                for j in range(i + 1, dimension):
                    dist_matrix[i][j] = distances[idx]
                    dist_matrix[j][i] = distances[idx]
                    idx += 1
        elif edge_weight_format == 'LOWER_DIAG_ROW':
            idx = 0
            for i in range(dimension):
                for j in range(i + 1):
                    dist_matrix[i][j] = distances[idx]
                    if i != j:
                        dist_matrix[j][i] = distances[idx]
                    idx += 1
        
        # Generate dummy coordinates for visualization
        coords = np.random.rand(dimension, 2) * 100
        
        return dist_matrix, coords

class TSPSolver:
    """Base class for TSP solving algorithms"""
    
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
    
    def calculate_tour_length(self, tour):
        """Calculate total length of a tour"""
        length = 0
        for i in range(len(tour)):
            length += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return length
    
    def generate_random_tour(self):
        """Generate a random tour"""
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour

class NBGA(TSPSolver):
    """Neighbourhood-based Genetic Algorithm"""
    
    def __init__(self, distance_matrix, pop_size=100, generations=500, 
                 mutation_rate=0.02, elite_size=20):
        super().__init__(distance_matrix)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
    
    def solve(self):
        """Solve TSP using NBGA"""
        # Initialize population
        population = [self.generate_random_tour() for _ in range(self.pop_size)]
        
        best_distance = float('inf')
        best_tour = None
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [(tour, self.calculate_tour_length(tour)) 
                            for tour in population]
            fitness_scores.sort(key=lambda x: x[1])
            
            # Update best solution
            if fitness_scores[0][1] < best_distance:
                best_distance = fitness_scores[0][1]
                best_tour = fitness_scores[0][0][:]
            
            # Select elite
            elite = [tour for tour, _ in fitness_scores[:self.elite_size]]
            
            # Create new population
            new_population = elite[:]
            
            while len(new_population) < self.pop_size:
                # Tournament selection
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                
                # Neighbourhood-based crossover
                child = self._neighbourhood_crossover(parent1, parent2)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._neighbourhood_mutation(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_tour, best_distance
    
    def _tournament_selection(self, fitness_scores, tournament_size=5):
        """Tournament selection"""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        return min(tournament, key=lambda x: x[1])[0]
    
    def _neighbourhood_crossover(self, parent1, parent2):
        """Neighbourhood-based crossover operator"""
        child = [-1] * self.n_cities
        
        # Select a random segment from parent1
        start = random.randint(0, self.n_cities - 1)
        end = random.randint(start, self.n_cities - 1)
        
        # Copy segment from parent1
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # Fill remaining positions based on neighbourhood similarity
        remaining = [city for city in parent2 if city not in child]
        
        for i in range(self.n_cities):
            if child[i] == -1:
                # Find closest city from remaining that maintains neighbourhood structure
                best_city = remaining[0]
                min_penalty = float('inf')
                
                for city in remaining:
                    penalty = self._calculate_neighbourhood_penalty(child, i, city)
                    if penalty < min_penalty:
                        min_penalty = penalty
                        best_city = city
                
                child[i] = best_city
                remaining.remove(best_city)
        
        return child
    
    def _calculate_neighbourhood_penalty(self, partial_tour, position, city):
        """Calculate penalty for placing city at position based on neighbourhood"""
        penalty = 0
        
        # Check left neighbor
        if position > 0 and partial_tour[position - 1] != -1:
            penalty += self.distance_matrix[partial_tour[position - 1]][city]
        
        # Check right neighbor
        if position < len(partial_tour) - 1 and partial_tour[position + 1] != -1:
            penalty += self.distance_matrix[city][partial_tour[position + 1]]
        
        return penalty
    
    def _neighbourhood_mutation(self, tour):
        """Neighbourhood-based mutation"""
        mutated_tour = tour[:]
        
        # Select two random positions
        i, j = random.sample(range(self.n_cities), 2)
        
        # Swap if it improves local neighbourhood
        original_cost = (self.distance_matrix[mutated_tour[i-1]][mutated_tour[i]] +
                        self.distance_matrix[mutated_tour[i]][mutated_tour[(i+1) % self.n_cities]] +
                        self.distance_matrix[mutated_tour[j-1]][mutated_tour[j]] +
                        self.distance_matrix[mutated_tour[j]][mutated_tour[(j+1) % self.n_cities]])
        
        # Try swap
        mutated_tour[i], mutated_tour[j] = mutated_tour[j], mutated_tour[i]
        
        new_cost = (self.distance_matrix[mutated_tour[i-1]][mutated_tour[i]] +
                   self.distance_matrix[mutated_tour[i]][mutated_tour[(i+1) % self.n_cities]] +
                   self.distance_matrix[mutated_tour[j-1]][mutated_tour[j]] +
                   self.distance_matrix[mutated_tour[j]][mutated_tour[(j+1) % self.n_cities]])
        
        # Revert if not improved
        if new_cost > original_cost:
            mutated_tour[i], mutated_tour[j] = mutated_tour[j], mutated_tour[i]
        
        return mutated_tour

class SWAP_GATSP(TSPSolver):
    """Genetic Algorithm with Swap Mutation for TSP"""
    
    def __init__(self, distance_matrix, pop_size=100, generations=500, 
                 mutation_rate=0.02, elite_size=20):
        super().__init__(distance_matrix)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
    
    def solve(self):
        population = [self.generate_random_tour() for _ in range(self.pop_size)]
        
        best_distance = float('inf')
        best_tour = None
        
        for generation in range(self.generations):
            fitness_scores = [(tour, self.calculate_tour_length(tour)) 
                            for tour in population]
            fitness_scores.sort(key=lambda x: x[1])
            
            if fitness_scores[0][1] < best_distance:
                best_distance = fitness_scores[0][1]
                best_tour = fitness_scores[0][0][:]
            
            elite = [tour for tour, _ in fitness_scores[:self.elite_size]]
            new_population = elite[:]
            
            while len(new_population) < self.pop_size:
                parent1 = self._roulette_selection(fitness_scores)
                parent2 = self._roulette_selection(fitness_scores)
                
                child = self._pmx_crossover(parent1, parent2)
                
                if random.random() < self.mutation_rate:
                    child = self._swap_mutation(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_tour, best_distance
    
    def _roulette_selection(self, fitness_scores):
        """Roulette wheel selection"""
        max_fitness = max(score for _, score in fitness_scores)
        adjusted_fitness = [max_fitness - score + 1 for _, score in fitness_scores]
        total_fitness = sum(adjusted_fitness)
        
        r = random.uniform(0, total_fitness)
        cumulative = 0
        
        for i, fitness in enumerate(adjusted_fitness):
            cumulative += fitness
            if cumulative >= r:
                return fitness_scores[i][0]
        
        return fitness_scores[-1][0]
    
    def _pmx_crossover(self, parent1, parent2):
        """Partially Matched Crossover"""
        size = len(parent1)
        child = [-1] * size
        
        # Select crossover points
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)
        
        # Copy segment from parent1
        child[start:end + 1] = parent1[start:end + 1]
        
        # Fill remaining positions
        for i in range(size):
            if child[i] == -1:
                candidate = parent2[i]
                while candidate in child:
                    idx = parent1.index(candidate)
                    candidate = parent2[idx]
                child[i] = candidate
        
        return child
    
    def _swap_mutation(self, tour):
        """Simple swap mutation"""
        mutated_tour = tour[:]
        i, j = random.sample(range(self.n_cities), 2)
        mutated_tour[i], mutated_tour[j] = mutated_tour[j], mutated_tour[i]
        return mutated_tour

class OX_SIM(TSPSolver):
    """Order Crossover with Simulated Annealing"""
    
    def __init__(self, distance_matrix, pop_size=100, generations=500, 
                 initial_temp=1000, cooling_rate=0.995):
        super().__init__(distance_matrix)
        self.pop_size = pop_size
        self.generations = generations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def solve(self):
        population = [self.generate_random_tour() for _ in range(self.pop_size)]
        temperature = self.initial_temp
        
        best_distance = float('inf')
        best_tour = None
        
        for generation in range(self.generations):
            fitness_scores = [(tour, self.calculate_tour_length(tour)) 
                            for tour in population]
            fitness_scores.sort(key=lambda x: x[1])
            
            if fitness_scores[0][1] < best_distance:
                best_distance = fitness_scores[0][1]
                best_tour = fitness_scores[0][0][:]
            
            new_population = []
            
            for _ in range(self.pop_size):
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                
                child = self._order_crossover(parent1, parent2)
                child = self._simulated_annealing_local_search(child, temperature)
                
                new_population.append(child)
            
            population = new_population
            temperature *= self.cooling_rate
        
        return best_tour, best_distance
    
    def _order_crossover(self, parent1, parent2):
        """Order Crossover (OX)"""
        size = len(parent1)
        child = [-1] * size
        
        # Select crossover segment
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)
        
        # Copy segment from parent1
        child[start:end + 1] = parent1[start:end + 1]
        
        # Fill remaining positions from parent2
        remaining = [city for city in parent2 if city not in child]
        j = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
        
        return child
    
    def _simulated_annealing_local_search(self, tour, temperature):
        """Apply simulated annealing for local improvement"""
        current_tour = tour[:]
        current_distance = self.calculate_tour_length(current_tour)
        
        for _ in range(10):  # Limited iterations for efficiency
            # Generate neighbor by 2-opt
            new_tour = self._two_opt(current_tour)
            new_distance = self.calculate_tour_length(new_tour)
            
            # Accept or reject based on simulated annealing
            if (new_distance < current_distance or 
                random.random() < math.exp(-(new_distance - current_distance) / temperature)):
                current_tour = new_tour
                current_distance = new_distance
        
        return current_tour
    
    def _two_opt(self, tour):
        """2-opt local search move"""
        new_tour = tour[:]
        i, j = sorted(random.sample(range(len(tour)), 2))
        new_tour[i:j+1] = reversed(new_tour[i:j+1])
        return new_tour

class MOC_SIM(TSPSolver):
    """Modified Order Crossover with Simulated Annealing"""
    
    def __init__(self, distance_matrix, pop_size=100, generations=500, 
                 initial_temp=1000, cooling_rate=0.995):
        super().__init__(distance_matrix)
        self.pop_size = pop_size
        self.generations = generations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def solve(self):
        population = [self.generate_random_tour() for _ in range(self.pop_size)]
        temperature = self.initial_temp
        
        best_distance = float('inf')
        best_tour = None
        
        for generation in range(self.generations):
            fitness_scores = [(tour, self.calculate_tour_length(tour)) 
                            for tour in population]
            fitness_scores.sort(key=lambda x: x[1])
            
            if fitness_scores[0][1] < best_distance:
                best_distance = fitness_scores[0][1]
                best_tour = fitness_scores[0][0][:]
            
            new_population = []
            
            for _ in range(self.pop_size):
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                
                child = self._modified_order_crossover(parent1, parent2)
                child = self._simulated_annealing_optimization(child, temperature)
                
                new_population.append(child)
            
            population = new_population
            temperature *= self.cooling_rate
        
        return best_tour, best_distance
    
    def _tournament_selection(self, fitness_scores, tournament_size=3):
        """Tournament selection"""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        return min(tournament, key=lambda x: x[1])[0]
    
    def _modified_order_crossover(self, parent1, parent2):
        """Modified Order Crossover with distance consideration"""
        size = len(parent1)
        child = [-1] * size
        
        # Select multiple smaller segments based on distance
        segments = []
        segment_length = max(2, size // 4)
        
        for _ in range(2):  # Two segments
            start = random.randint(0, size - segment_length)
            end = min(start + segment_length - 1, size - 1)
            segments.append((start, end))
        
        # Copy segments from parent1
        for start, end in segments:
            child[start:end + 1] = parent1[start:end + 1]
        
        # Fill remaining positions considering distance
        remaining = [city for city in parent2 if city not in child]
        
        for i in range(size):
            if child[i] == -1:
                if remaining:
                    # Choose city that minimizes local distance
                    best_city = remaining[0]
                    min_cost = float('inf')
                    
                    for city in remaining:
                        cost = 0
                        if i > 0 and child[i-1] != -1:
                            cost += self.distance_matrix[child[i-1]][city]
                        if i < size - 1 and child[i+1] != -1:
                            cost += self.distance_matrix[city][child[i+1]]
                        
                        if cost < min_cost:
                            min_cost = cost
                            best_city = city
                    
                    child[i] = best_city
                    remaining.remove(best_city)
        
        return child
    
    def _simulated_annealing_optimization(self, tour, temperature):
        """Enhanced simulated annealing with multiple operators"""
        current_tour = tour[:]
        current_distance = self.calculate_tour_length(current_tour)
        
        for _ in range(15):  # More iterations for better optimization
            # Randomly choose optimization operator
            if random.random() < 0.5:
                new_tour = self._two_opt(current_tour)
            else:
                new_tour = self._or_opt(current_tour)
            
            new_distance = self.calculate_tour_length(new_tour)
            
            # Simulated annealing acceptance
            if (new_distance < current_distance or 
                random.random() < math.exp(-(new_distance - current_distance) / max(temperature, 0.1))):
                current_tour = new_tour
                current_distance = new_distance
        
        return current_tour
    
    def _two_opt(self, tour):
        """2-opt improvement"""
        new_tour = tour[:]
        i, j = sorted(random.sample(range(len(tour)), 2))
        new_tour[i:j+1] = reversed(new_tour[i:j+1])
        return new_tour
    
    def _or_opt(self, tour):
        """Or-opt improvement (relocate segment)"""
        new_tour = tour[:]
        size = len(tour)
        
        # Select a segment to relocate
        segment_length = random.randint(1, 3)
        start = random.randint(0, size - segment_length)
        end = start + segment_length - 1
        
        # Remove segment
        segment = new_tour[start:end + 1]
        remaining = new_tour[:start] + new_tour[end + 1:]
        
        # Insert at new position
        new_pos = random.randint(0, len(remaining))
        new_tour = remaining[:new_pos] + segment + remaining[new_pos:]
        
        return new_tour


def run_algorithms(dataset_dir):
    datasets = ['bayg29.tsp.gz', 'gr24.tsp.gz', 'st70.tsp.gz']
    datasets_alt = ['bayg29.tsp', 'gr24.tsp', 'st70.tsp']

    algorithms = {
        'NBGA': NBGA,
        'SWAP_GATSP': SWAP_GATSP,
        'OX_SIM': OX_SIM,
        'MOC_SIM': MOC_SIM
    }

    results = {alg: [] for alg in algorithms}
    dataset_names = []

    for dataset, alt_dataset in zip(datasets, datasets_alt):
        dataset_path = os.path.join(dataset_dir, dataset)
        alt_dataset_path = os.path.join(dataset_dir, alt_dataset)
        try:
            if Path(dataset_path).exists():
                distance_matrix, coords = TSPParser.parse_tsp_file(dataset_path)
                dataset_name = dataset.replace('.tsp.gz', '')
            elif Path(alt_dataset_path).exists():
                distance_matrix, coords = TSPParser.parse_tsp_file(alt_dataset_path)
                dataset_name = alt_dataset.replace('.tsp', '')
            else:
                print(f"Dataset {dataset} not found in {dataset_dir}, skipping...")
                continue
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
            continue

        dataset_names.append(dataset_name)
        print(f"Testing on {dataset_name} ({len(distance_matrix)} cities)")

        for alg_name, alg_class in algorithms.items():
            print(f"Running {alg_name}...")

            pop_size = min(100, max(50, len(distance_matrix) * 2))
            generations = min(500, max(200, len(distance_matrix) * 5))

            start_time = time.time()

            solver = alg_class(distance_matrix, pop_size=pop_size, generations=generations)
            best_tour, best_distance = solver.solve()
            end_time = time.time()

            results[alg_name].append({
                'distance': float(best_distance),
                'time': float(end_time - start_time),
                'tour': best_tour
            })

            print(f"{alg_name}: Distance = {best_distance:.2f}, Time = {end_time - start_time:.2f}s")

    return results, dataset_names, list(algorithms.keys())

def create_comparison_plot(results, dataset_names, algorithm_names):
    plt.figure(figsize=(12, 8))

    x = np.arange(len(dataset_names))
    width = 0.2

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (alg_name, color) in enumerate(zip(algorithm_names, colors)):
        distances = [results[alg_name][j]['distance'] for j in range(len(dataset_names))]
        plt.bar(x + i * width, distances, width, label=alg_name, color=color, alpha=0.8)

    plt.xlabel('Datasets', fontsize=12)
    plt.ylabel('Best Tour Distance', fontsize=12)
    plt.title('TSP Algorithm Comparison: Best Tour Distance by Dataset', fontsize=14, fontweight='bold')
    plt.xticks(x + width * 1.5, dataset_names)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)

    for i, alg_name in enumerate(algorithm_names):
        distances = [results[alg_name][j]['distance'] for j in range(len(dataset_names))]
        for j, distance in enumerate(distances):
            plt.text(j + i * width, distance + max(distances) * 0.01, f'{distance:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

def main():
    dataset_dir = "./tsp_dataset"  # You can change this path as needed
    random.seed(42)
    np.random.seed(42)

    start_time = time.time()
    try:
        results, dataset_names, algorithm_names = run_algorithms(dataset_dir)
        total_time = time.time() - start_time

        print(f"Comparison completed in {total_time:.2f} seconds!")
        create_comparison_plot(results, dataset_names, algorithm_names)

        print("\nResults Summary:")
        for i, dataset_name in enumerate(dataset_names):
            print(f"\nDataset: {dataset_name}")
            for alg_name in algorithm_names:
                distance = results[alg_name][i]['distance']
                exec_time = results[alg_name][i]['time']
                print(f"  {alg_name}: Distance = {distance:.2f}, Time = {exec_time:.2f}s")

        print("\nAlgorithm Ranking by Average Distance:")
        avg_performance = {}
        for alg in algorithm_names:
            avg_distance = np.mean([results[alg][i]['distance'] for i in range(len(dataset_names))])
            avg_time = np.mean([results[alg][i]['time'] for i in range(len(dataset_names))])
            avg_performance[alg] = (avg_distance, avg_time)

        sorted_algs = sorted(avg_performance.items(), key=lambda x: x[1][0])
        for rank, (alg, (avg_dist, avg_time)) in enumerate(sorted_algs, 1):
            print(f"  {rank}. {alg}: Avg Distance = {avg_dist:.2f}, Avg Time = {avg_time:.2f}s")

    except FileNotFoundError:
        print(f"Dataset directory '{dataset_dir}' not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
