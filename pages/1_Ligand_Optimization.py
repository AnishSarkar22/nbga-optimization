# NBGA Fitness Evolution in Ligand Optimization script
# Replicates Table 3 and Figure 11 results from the research paper

import random
import matplotlib.pyplot as plt
import numpy as np
import math
import streamlit as st

# Constants from PDF
# C_n = 10000 
# C_m = 500
C_n = 1e5 # see README.md
C_m = 200 # see README.md
n = 12 # Van der Waals exponent
m = 6 # Van der Waals exponent
K = 100 # Fitness scaling constant

# Functional group bond lengths (Table 2 from PDF)
BOND_LENGTHS = {
    1: 0.65, 2: 1.75, 3: 1.1, 4: 2.2,
    5: 0.01, 6: 1.9, 7: 2.7, 8: 0.0
}

# Protein residue coordinates
RESIDUES = [
    (2.0, 3.0), (4.0, 2.5), (6.0, 3.2), (8.0, 2.8),
    (3.5, 5.0), (5.5, 4.8), (7.5, 5.2), (4.5, 1.0)
]

def calculate_coordinates(chromosome, is_right_tree=True):
    """
    Algorithm: Coordinate Mapping Algorithm
    Purpose: Converts chromosome representation to 2D spatial coordinates
    
    This algorithm maps each functional group in the chromosome to a 2D position
    based on bond lengths and a simple geometric arrangement. It implements a
    linear chain model where each group is positioned sequentially along the x-axis
    with slight vertical variations to create a realistic molecular structure.
    """
    coordinates = []
    base_x = 0.0 if is_right_tree else 0.0
    
    for i, group in enumerate(chromosome):
        if group == 8:
            continue  # Skip empty positions (group 8 represents empty space)
        # Linear chain positioning: x increases by bond length for each group
        x = base_x + BOND_LENGTHS[group] * (i + 1)
        # Simple vertical arrangement: y varies in a pattern to create 3D-like structure
        y = 2.0 + (i % 3) * 0.5
        coordinates.append((x, y))
    
    return coordinates

def vdw_energy(r):
    """
    Algorithm: Modified Lennard-Jones Potential
    Purpose: Calculates van der Waals interaction energy between atoms
    
    This implements a modified Lennard-Jones 12-6 potential:
    V(r) = C_n/r^12 - C_m/r^6
    
    The algorithm includes:
    1. Repulsive term (r^12): Prevents atoms from overlapping
    2. Attractive term (r^6): Represents dispersion forces
    3. Hard cutoff: Energy penalty for distances outside optimal range (0.7-2.7 Å)
    """
    if r < 0.7 or r > 2.7:
        return 1000  # Hard cutoff penalty for distances outside optimal range
    return (C_n / (r**n)) - (C_m / (r**m))

def compute_interaction_energy(chromosome):
    """
    Algorithm: Minimum Energy Interaction Algorithm
    Purpose: Calculates total interaction energy between ligand and protein
    
    This algorithm implements:
    1. Coordinate mapping: Convert chromosome to spatial coordinates
    2. Nearest neighbor search: Find closest protein residue for each ligand group
    3. Energy minimization: Calculate minimum energy interaction for each group
    4. Energy summation: Sum all interactions to get total binding energy
    
    The algorithm assumes each ligand group interacts with the nearest protein residue,
    which is a common approximation in molecular docking.
    """
    coords = calculate_coordinates(chromosome)
    empty_count = chromosome.count(8)
    # Penalize if more than half the ligand is empty
    if len(coords) == 0 or empty_count > len(chromosome) // 2:
        return 1e6
    total_energy = 0
    
    for group_coord in coords:
        min_energy = float('inf')
        # Nearest neighbor search: find closest protein residue
        for residue_coord in RESIDUES:
            # Euclidean distance calculation in 2D space
            distance = math.sqrt((group_coord[0] - residue_coord[0])**2 + 
                               (group_coord[1] - residue_coord[1])**2)
            energy = vdw_energy(distance)
            min_energy = min(min_energy, energy)  # Energy minimization
        total_energy += min_energy
    
    return total_energy

def create_ring_topology(population):
    """
    Algorithm: Ring Topology Algorithm
    Purpose: Creates neighborhood structure for NBGA
    
    This algorithm implements a ring topology where individuals are arranged
    in a circular manner. The shuffled arrangement ensures that:
    1. Each individual has exactly two neighbors (except in edge cases)
    2. The neighborhood structure is randomized each generation
    3. Local interactions are maintained while preventing premature convergence
    
    This is a key feature of NBGA that differentiates it from standard GA.
    """
    shuffled = population.copy()
    random.shuffle(shuffled)  # Randomize neighborhood connections
    return shuffled

def trio_selection(parent, child1, child2):
    """
    Algorithm: Tournament Selection (Trio Variant)
    Purpose: Selects the best individual from a parent-offspring trio
    
    This implements a tournament selection algorithm where:
    1. Three candidates compete: parent and two offspring
    2. Fitness evaluation: Calculate energy for each candidate
    3. Winner selection: Choose the individual with minimum energy (best fitness)
    
    This selection method provides:
    - Elitism: Best solutions are preserved
    - Diversity: New solutions can replace parents if better
    - Local competition: Selection pressure within neighborhoods
    """
    candidates = [parent, child1, child2]
    energies = [compute_interaction_energy(c) for c in candidates]
    best_idx = energies.index(min(energies))  # Find minimum energy (best fitness)
    return candidates[best_idx]

def crossover(parent1, parent2):
    """
    Algorithm: Single-Point Crossover
    Purpose: Creates offspring by combining genetic material from parents
    
    This genetic algorithm crossover operator:
    1. Safety check: Ensures parents have same length
    2. Crossover point selection: Random position between 1 and length-1
    3. Genetic recombination: Exchange genetic material at crossover point
    4. Offspring generation: Create two complementary offspring
    
    The algorithm maintains genetic diversity while preserving good building blocks.
    """
    if len(parent1) != len(parent2):
        min_len = min(len(parent1), len(parent2))
        parent1, parent2 = parent1[:min_len], parent2[:min_len]
    
    point = random.randint(1, len(parent1) - 1)  # Random crossover point
    child1 = parent1[:point] + parent2[point:]    # First offspring
    child2 = parent2[:point] + parent1[point:]    # Second offspring
    return child1, child2

def mutate(chromosome, rate=0.1):
    """
    Algorithm: Uniform Random Mutation
    Purpose: Introduces genetic diversity through random changes
    
    This mutation algorithm:
    1. Gene-by-gene mutation: Each gene has independent mutation probability
    2. Uniform random replacement: Mutated genes get random values (1-8)
    3. Rate control: Mutation rate determines exploration vs exploitation balance
    
    Mutation prevents premature convergence and maintains population diversity.
    """
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if random.random() < rate:  # Mutation probability check
            mutated[i] = random.randint(1, 8)  # Random gene replacement
    return mutated

def rolling_average(data, window_size=5):
    """
    Algorithm: Moving Average Filter (Convolution-based)
    Purpose: Smooths noisy data to reveal underlying trends
    
    This signal processing algorithm:
    1. Convolution operation: Uses numpy's convolve function
    2. Window averaging: Each point is averaged with its neighbors
    3. Edge handling: 'valid' mode returns only fully computed points
    
    The algorithm reduces noise in the energy evolution curve to better
    visualize convergence patterns and trends.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def nbga_ligand_optimization(pop_size=50, generations=100, tree_size=10):
    """
    Algorithm: Neighborhood-Based Genetic Algorithm (NBGA)
    Purpose: Optimizes ligand structures for protein binding
    
    This is the main evolutionary algorithm that combines:
    
    1. Population Initialization:
       - Random chromosome generation
       - Each chromosome represents a ligand structure
    
    2. Evolutionary Loop:
       - Ring topology creation for neighborhood structure
       - Pairwise parent selection from neighborhoods
       - Crossover and mutation for offspring generation
       - Trio selection for survival competition
       - Population replacement
    
    3. Convergence Monitoring:
       - Energy tracking across generations
       - Best solution identification
    
    The algorithm balances exploration (mutation) and exploitation (selection)
    to find optimal ligand structures with minimum binding energy.
    """
    random.seed(42)  # For reproducibility
    
    # Population initialization algorithm
    population = []
    for _ in range(pop_size):
        chromosome = [random.randint(1, 8) for _ in range(tree_size)]
        population.append(chromosome)
    
    best_energies = []
    
    # Main evolutionary loop
    for gen in range(generations):
        # Ring topology algorithm
        ring_parents = create_ring_topology(population)
        new_population = []
        
        # Neighborhood-based evolution
        for i in range(0, len(ring_parents), 2):
            p1 = ring_parents[i]
            p2 = ring_parents[(i + 1) % len(ring_parents)]
            
            # Genetic operators
            c1, c2 = crossover(p1, p2)  # Crossover algorithm
            c1 = mutate(c1)             # Mutation algorithm
            c2 = mutate(c2)
            
            # Selection algorithm
            selected1 = trio_selection(p1, c1, c2)
            selected2 = trio_selection(p2, c1, c2)
            
            new_population.extend([selected1, selected2])
        
        # Population replacement
        population = new_population[:pop_size]
        
        # Fitness evaluation and monitoring
        best_energy = min(compute_interaction_energy(ind) for ind in population)
        best_energies.append(best_energy)
        
        if gen % 20 == 0:
            print(f"Generation {gen}: Best Energy = {best_energy:.4f} Kcal/mol")
    
    # Final solution selection
    best_individual = min(population, key=compute_interaction_energy)
    return best_individual, best_energies

def main():
    st.title("NBGA Ligand Optimization")
    st.write("Neighborhood-Based Genetic Algorithm for Ligand-Protein Binding")

    pop_size = st.sidebar.slider("Population Size", 10, 200, 50, step=5)
    generations = st.sidebar.slider("Generations", 10, 500, 100, step=10)
    tree_size = st.sidebar.slider("Ligand Tree Size", 5, 20, 10, step=1)
    window_size = st.sidebar.slider("Smoothing Window Size", 1, 20, 5, step=1)

    if st.button("Run NBGA Optimization"):
        st.write("Running NBGA for Ligand Optimization...")
        best_ligand, energy_progress = nbga_ligand_optimization(
            pop_size=pop_size, generations=generations, tree_size=tree_size
        )
        smoothed_energy = rolling_average(energy_progress, window_size=window_size)

        st.write(f"**Best Ligand:** {best_ligand}")
        st.write(f"**Best Energy:** {compute_interaction_energy(best_ligand):.4f} Kcal/mol")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 7))
        plt.plot(energy_progress, alpha=0.4, color='darkblue', linewidth=1, label="Raw Energy")
        plt.plot(range(len(smoothed_energy)), smoothed_energy, color='red', linewidth=2, label="Smoothed Energy")
        plt.xlabel("Generation")
        plt.ylabel("Interaction Energy (Kcal/mol)")
        plt.title("NBGA Ligand Optimization - Energy Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    st.markdown("---")
    st.write("Adjust parameters in the sidebar and click **Run NBGA Optimization** to start.")

# Run with your smoothing visualization
if __name__ == "__main__":
    main()