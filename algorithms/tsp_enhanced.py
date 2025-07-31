# Neighborhood-Based Genetic Algorithm (NBGA) comparison with SWAP_GATSP, OX_SIM, and MOC_SIM applied to the Traveling Salesman Problem (TSP)


# This enhanced version introduces several improvements over the original tsp.py:
# - NBGA uses advanced initialization (nearest neighbor heuristic), adaptive mutation rates, and diverse elite selection.
# - Crossover and mutation operators are more sophisticated, including multi-point neighbourhood crossover, multi-operator mutation (2-opt, 3-opt, Or-opt, Lin-Kernighan), and intensive local search.
# - Population and generations are adaptively set based on problem size.
# - All algorithms (NBGA, SWAP_GATSP, OX_SIM, MOC_SIM) are implemented with enhanced heuristics and metaheuristics.
# - Comprehensive performance comparison and visualization are included, with detailed statistical summaries.


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
    
    # for st70.tsp file
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
        
        if edge_weight_format == 'UPPER_ROW':           # for bayg29.tsp file
            idx = 0
            for i in range(dimension):
                for j in range(i + 1, dimension):
                    dist_matrix[i][j] = distances[idx]
                    dist_matrix[j][i] = distances[idx]
                    idx += 1
        elif edge_weight_format == 'LOWER_DIAG_ROW':    # for gr24.tsp file
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
    """Enhanced Neighbourhood-based Genetic Algorithm"""
    
    def __init__(self, distance_matrix, pop_size=100, generations=500, 
                 mutation_rate=0.15, elite_size=30, crossover_rate=0.9):  # Increased mutation and elite size
        super().__init__(distance_matrix)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
    
    def solve(self):
        # Enhanced initialization with nearest neighbor heuristic
        population = []
        
        # Add 30% nearest neighbor solutions
        for _ in range(int(0.3 * self.pop_size)):
            start_city = random.randint(0, self.n_cities - 1)
            nn_tour = self._nearest_neighbor_heuristic(start_city)
            population.append(nn_tour)
        
        # Added random solutions
        for _ in range(self.pop_size - len(population)):
            population.append(self.generate_random_tour())
        
        best_distance = float('inf')
        best_tour = None
        stagnation_counter = 0
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [(tour, self.calculate_tour_length(tour)) 
                            for tour in population]
            fitness_scores.sort(key=lambda x: x[1])
            
            # Update best solution
            if fitness_scores[0][1] < best_distance:
                best_distance = fitness_scores[0][1]
                best_tour = fitness_scores[0][0][:]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Adaptive mutation rate based on stagnation
            adaptive_mutation_rate = self.mutation_rate
            if stagnation_counter > 50:
                adaptive_mutation_rate = min(0.3, self.mutation_rate * 2)
            
            # Select elite with diversity consideration
            elite = self._diverse_elite_selection(fitness_scores)
            
            # Create new population
            new_population = elite[:]
            
            while len(new_population) < self.pop_size:
                if random.random() < self.crossover_rate:
                    # Enhanced tournament selection
                    parent1 = self._tournament_selection(fitness_scores, tournament_size=7)
                    parent2 = self._tournament_selection(fitness_scores, tournament_size=7)
                    
                    # Multi-point neighbourhood crossover
                    child = self._enhanced_neighbourhood_crossover(parent1, parent2)
                else:
                    # Local search on elite solution
                    child = self._local_search_mutation(random.choice(elite))
                
                # Enhanced mutation with multiple operators
                if random.random() < adaptive_mutation_rate:
                    child = self._multi_operator_mutation(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_tour, best_distance
    
    def _nearest_neighbor_heuristic(self, start_city):
        """Generate tour using nearest neighbor heuristic"""
        tour = [start_city]
        remaining = set(range(self.n_cities)) - {start_city}
        
        current_city = start_city
        while remaining:
            nearest_city = min(remaining, 
                             key=lambda city: self.distance_matrix[current_city][city])
            tour.append(nearest_city)
            remaining.remove(nearest_city)
            current_city = nearest_city
        
        return tour
    
    def _diverse_elite_selection(self, fitness_scores):
        """Select elite with diversity consideration"""
        elite = []
        candidates = fitness_scores[:self.elite_size * 2]  # Consider more candidates
        
        # Always include best solution
        elite.append(candidates[0][0])
        
        # Select diverse solutions
        while len(elite) < self.elite_size and len(candidates) > len(elite):
            best_candidate = None
            max_diversity = -1
            
            for candidate_tour, _ in candidates[len(elite):]:
                # Calculate diversity as minimum hamming distance to elite
                min_distance = min(self._hamming_distance(candidate_tour, elite_tour) 
                                 for elite_tour in elite)
                
                if min_distance > max_diversity:
                    max_diversity = min_distance
                    best_candidate = candidate_tour
            
            if best_candidate:
                elite.append(best_candidate)
            else:
                elite.append(candidates[len(elite)][0])
        
        return elite
    
    def _hamming_distance(self, tour1, tour2):
        """Calculate Hamming distance between two tours"""
        return sum(1 for i in range(len(tour1)) if tour1[i] != tour2[i])
    
    def _tournament_selection(self, fitness_scores, tournament_size=7):
        """Tournament selection method"""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        return min(tournament, key=lambda x: x[1])[0]
    
    def _enhanced_neighbourhood_crossover(self, parent1, parent2):
        """Enhanced multi-point neighbourhood crossover"""
        child = [-1] * self.n_cities
        
        # Multiple segments with variable sizes
        num_segments = random.randint(2, 4)
        segments = []
        
        for _ in range(num_segments):
            start = random.randint(0, self.n_cities - 1)
            length = random.randint(2, max(3, self.n_cities // 6))
            end = min(start + length - 1, self.n_cities - 1)
            segments.append((start, end))
        
        # Copy segments from parent1
        for start, end in segments:
            for i in range(start, end + 1):
                if child[i] == -1:
                    child[i] = parent1[i]
        
        # Fill remaining with enhanced neighbourhood logic
        remaining = [city for city in parent2 if city not in child]
        
        for i in range(self.n_cities):
            if child[i] == -1:
                if remaining:
                    best_city = self._select_best_neighbour_city(child, i, remaining)
                    child[i] = best_city
                    remaining.remove(best_city)
        
        return child
    
    def _select_best_neighbour_city(self, partial_tour, position, candidates):
        """Select best city considering neighbourhood and global structure"""
        if not candidates:
            return candidates[0]
        
        best_city = candidates[0]
        min_cost = float('inf')
        
        for city in candidates:
            cost = 0
            weight_factor = 1.0
            
            # Local neighbourhood cost
            if position > 0 and partial_tour[position - 1] != -1:
                cost += self.distance_matrix[partial_tour[position - 1]][city] * weight_factor
            
            if position < len(partial_tour) - 1 and partial_tour[position + 1] != -1:
                cost += self.distance_matrix[city][partial_tour[position + 1]] * weight_factor
            
            # Extended neighbourhood cost (look ahead/behind 2 positions)
            if position > 1 and partial_tour[position - 2] != -1:
                cost += self.distance_matrix[partial_tour[position - 2]][city] * 0.3
            
            if position < len(partial_tour) - 2 and partial_tour[position + 2] != -1:
                cost += self.distance_matrix[city][partial_tour[position + 2]] * 0.3
            
            # Penalize already used edges
            edge_penalty = 0
            for used_city in partial_tour:
                if used_city != -1 and used_city != city:
                    edge_penalty += 1.0 / (1.0 + self.distance_matrix[city][used_city])
            
            total_cost = cost + edge_penalty * 0.1
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_city = city
        
        return best_city
    
    def _multi_operator_mutation(self, tour):
        """Apply multiple mutation operators"""
        mutated_tour = tour[:]
        
        # Choose mutation operator based on probability
        rand = random.random()
        
        if rand < 0.4:  # 2-opt with multiple attempts
            mutated_tour = self._intensive_two_opt(mutated_tour)
        elif rand < 0.6:  # 3-opt
            mutated_tour = self._three_opt(mutated_tour)
        elif rand < 0.8:  # Or-opt
            mutated_tour = self._or_opt_improved(mutated_tour)
        else:  # Lin-Kernighan style move
            mutated_tour = self._lin_kernighan_move(mutated_tour)
        
        return mutated_tour
    
    def _intensive_two_opt(self, tour):
        """Intensive 2-opt with multiple random attempts"""
        best_tour = tour[:]
        best_distance = self.calculate_tour_length(best_tour)
        
        for attempt in range(5):  # Multiple attempts
            current_tour = tour[:]
            
            for _ in range(3):  # Multiple 2-opt moves per attempt
                i, j = sorted(random.sample(range(1, len(tour) - 1), 2))
                if j - i > 1:
                    new_tour = current_tour[:]
                    new_tour[i:j+1] = reversed(new_tour[i:j+1])
                    
                    if self.calculate_tour_length(new_tour) < self.calculate_tour_length(current_tour):
                        current_tour = new_tour
            
            if self.calculate_tour_length(current_tour) < best_distance:
                best_tour = current_tour
                best_distance = self.calculate_tour_length(current_tour)
        
        return best_tour
    
    def _three_opt(self, tour):
        """3-opt local search move"""
        n = len(tour)
        best_tour = tour[:]
        
        # Select three edges to break
        indices = sorted(random.sample(range(n), 3))
        i, j, k = indices
        
        # Try different reconnection patterns
        reconnections = [
            tour[:i] + tour[j:k+1] + tour[i:j] + tour[k+1:],
            tour[:i] + tour[j:k+1][::-1] + tour[i:j] + tour[k+1:],
            tour[:i] + tour[i:j][::-1] + tour[j:k+1] + tour[k+1:],
        ]
        
        best_distance = self.calculate_tour_length(best_tour)
        for new_tour in reconnections:
            new_distance = self.calculate_tour_length(new_tour)
            if new_distance < best_distance:
                best_tour = new_tour
                best_distance = new_distance
        
        return best_tour
    
    def _or_opt_improved(self, tour):
        """Improved Or-opt with multiple segment sizes"""
        best_tour = tour[:]
        best_distance = self.calculate_tour_length(best_tour)
        
        for segment_length in [1, 2, 3]:
            if segment_length >= len(tour):
                continue
                
            # Try multiple random moves
            for _ in range(3):
                start = random.randint(0, len(tour) - segment_length)
                
                # Extract segment
                segment = tour[start:start + segment_length]
                remaining = tour[:start] + tour[start + segment_length:]
                
                # Try different insertion positions
                for insert_pos in random.sample(range(len(remaining) + 1), 
                                               min(5, len(remaining) + 1)):
                    new_tour = remaining[:insert_pos] + segment + remaining[insert_pos:]
                    new_distance = self.calculate_tour_length(new_tour)
                    
                    if new_distance < best_distance:
                        best_tour = new_tour
                        best_distance = new_distance
        
        return best_tour
    
    def _lin_kernighan_move(self, tour):
        """Simplified Lin-Kernighan style move"""
        n = len(tour)
        best_tour = tour[:]
        
        # Select starting edge
        i = random.randint(0, n - 1)
        j = (i + 1) % n
        
        # Try to improve by breaking edge (i,j) and reconnecting
        for k in range(n):
            if k == i or k == j:
                continue
                
            next_k = (k + 1) % n
            if next_k == i or next_k == j:
                continue
            
            # Calculate improvement
            old_cost = (self.distance_matrix[tour[i]][tour[j]] + 
                       self.distance_matrix[tour[k]][tour[next_k]])
            new_cost = (self.distance_matrix[tour[i]][tour[k]] + 
                       self.distance_matrix[tour[j]][tour[next_k]])
            
            if new_cost < old_cost:
                # Perform reconnection
                if i < k:
                    new_tour = tour[:i+1] + tour[k:j:-1] + tour[k+1:]
                else:
                    new_tour = tour[:k+1] + tour[j:i:-1] + tour[j+1:]
                
                if len(new_tour) == n:
                    best_tour = new_tour
                break
        
        return best_tour
    
    def _local_search_mutation(self, tour):
        """Intensive local search on a tour"""
        current_tour = tour[:]
        
        # Apply multiple local search operators
        for _ in range(random.randint(2, 5)):
            operators = [self._intensive_two_opt, self._or_opt_local, self._swap_local]
            operator = random.choice(operators)
            current_tour = operator(current_tour)
        
        return current_tour
    
    def _or_opt_local(self, tour):
        """Local Or-opt improvement"""
        best_tour = tour[:]
        best_distance = self.calculate_tour_length(best_tour)
        
        for i in range(len(tour)):
            for j in range(len(tour)):
                if abs(i - j) <= 1:
                    continue
                
                new_tour = tour[:]
                city = new_tour.pop(i)
                new_tour.insert(j if j < i else j-1, city)
                
                new_distance = self.calculate_tour_length(new_tour)
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
        
        return best_tour
    
    def _swap_local(self, tour):
        """Local swap improvement"""
        best_tour = tour[:]
        best_distance = self.calculate_tour_length(best_tour)
        
        for _ in range(5):  # Multiple random swaps
            i, j = random.sample(range(len(tour)), 2)
            new_tour = tour[:]
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            
            new_distance = self.calculate_tour_length(new_tour)
            if new_distance < best_distance:
                best_tour = new_tour
                best_distance = new_distance
                tour = new_tour  # Update for next iteration
        
        return best_tour

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
    # Updated datasets list to include the new ones
    datasets = ['bayg29.tsp.gz', 'gr24.tsp.gz', 'st70.tsp.gz', 'eil51.tsp.gz', 'gr48.tsp.gz']
    datasets_alt = ['bayg29.tsp', 'gr24.tsp', 'st70.tsp', 'eil51.tsp', 'gr48.tsp']

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

            # Adaptive population size based on problem size
            pop_size = min(100, max(50, len(distance_matrix) * 2))
            
            # Adaptive generations based on problem complexity
            # if len(distance_matrix) <= 30:
            #     generations = 15000
            # elif len(distance_matrix) <= 50:
            #     generations = 20000
            # else:
            #     generations = 25000
            
            # Fixed generations for all datasets
            generations = 500

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
    """Create a comprehensive comparison plot for all 4 algorithms across all 5 datasets"""
    plt.figure(figsize=(16, 10))

    x = np.arange(len(dataset_names))
    width = 0.18  # Adjusted width for 4 algorithms

    # Enhanced color scheme for better distinction
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Create bars for each algorithm
    for i, (alg_name, color) in enumerate(zip(algorithm_names, colors)):
        distances = [results[alg_name][j]['distance'] for j in range(len(dataset_names))]
        bars = plt.bar(x + i * width, distances, width, 
                      label=alg_name, color=color, alpha=0.8, 
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars
        for j, (bar, distance) in enumerate(zip(bars, distances)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(distances) * 0.01, 
                    f'{distance:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xlabel('TSP Datasets', fontsize=14, fontweight='bold')
    plt.ylabel('Best Tour Distance', fontsize=14, fontweight='bold')
    plt.title('Comprehensive TSP Algorithm Comparison\nBest Tour Distance Across All Datasets', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Adjust x-axis labels
    plt.xticks(x + width * 1.5, dataset_names, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Enhanced legend
    plt.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add dataset size information as secondary x-axis labels
    dataset_sizes = []
    for name in dataset_names:
        if name == 'bayg29':
            dataset_sizes.append('(29 cities)')
        elif name == 'gr24':
            dataset_sizes.append('(24 cities)')
        elif name == 'st70':
            dataset_sizes.append('(70 cities)')
        elif name == 'eil51':
            dataset_sizes.append('(51 cities)')
        elif name == 'gr48':
            dataset_sizes.append('(48 cities)')
        else:
            dataset_sizes.append('')
    
    # Add secondary labels for city counts
    ax2 = plt.twiny()
    ax2.set_xlim(plt.xlim())
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(dataset_sizes, fontsize=10, style='italic')
    ax2.tick_params(axis='x', length=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Add a text box with summary statistics
    summary_text = "Algorithm Performance Summary:\n"
    avg_performance = {}
    for alg in algorithm_names:
        avg_distance = np.mean([results[alg][i]['distance'] for i in range(len(dataset_names))])
        avg_time = np.mean([results[alg][i]['time'] for i in range(len(dataset_names))])
        avg_performance[alg] = (avg_distance, avg_time)
    
    sorted_algs = sorted(avg_performance.items(), key=lambda x: x[1][0])
    for rank, (alg, (avg_dist, avg_time)) in enumerate(sorted_algs, 1):
        summary_text += f"{rank}. {alg}: {avg_dist:.1f} avg distance\n"
    
    plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.show()

    # Additional detailed performance analysis
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Performance by dataset
    for i, dataset_name in enumerate(dataset_names):
        print(f"\n{dataset_name.upper()} Dataset Results:")
        dataset_results = [(alg, results[alg][i]['distance'], results[alg][i]['time']) 
                          for alg in algorithm_names]
        dataset_results.sort(key=lambda x: x[1])  # Sort by distance
        
        for rank, (alg, dist, exec_time) in enumerate(dataset_results, 1):
            print(f"  {rank}. {alg:<12}: {dist:>8.2f} distance, {exec_time:>6.2f}s")
    
    # Statistical analysis
    print(f"\n{'STATISTICAL SUMMARY':<60}")
    print("-" * 60)
    for alg in algorithm_names:
        distances = [results[alg][i]['distance'] for i in range(len(dataset_names))]
        times = [results[alg][i]['time'] for i in range(len(dataset_names))]
        
        print(f"{alg:<12}: Avg={np.mean(distances):>7.1f}, "
              f"Std={np.std(distances):>6.1f}, "
              f"Min={np.min(distances):>7.1f}, "
              f"Max={np.max(distances):>7.1f}, "
              f"AvgTime={np.mean(times):>5.1f}s")

def main():
    dataset_dir = "../tsp_dataset/compressed"  # Change this path as needed
    random.seed(42)
    np.random.seed(42)

    start_time = time.time()
    try:
        results, dataset_names, algorithm_names = run_algorithms(dataset_dir)
        total_time = time.time() - start_time
        
        # uses optimal distances and finds out error for each algorithm
        optimal_distances = {
            'bayg29': 1610,
            'gr24': 1272,
            'gr48': 5046,
            'eil51': 426,
            'st70': 675
        }
        print("\nError Metrics Compared to Known Optimum:")
        for i, dataset_name in enumerate(dataset_names):
            if dataset_name in optimal_distances:
                optimal_distance = optimal_distances[dataset_name]
                print(f"\n{dataset_name} - Optimal: {optimal_distance}")
                for alg_name in results:
                    algorithm_distance = results[alg_name][i]['distance']
                    error_percentage = ((algorithm_distance - optimal_distance) / optimal_distance) * 100
                    print(f"  {alg_name}: {algorithm_distance:.2f} ({error_percentage:+.2f}% from optimal)")
        

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
