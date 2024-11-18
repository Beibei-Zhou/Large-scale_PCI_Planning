# main.py

import numpy as np
import random
from pci_planner import DataLoader, MatrixConstructor, FitnessCalculator
from evo_alg import EvolutionaryAlgorithm

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # File paths
    cell_info_file = '小区信息_filtered.xlsx'
    conflict_file = '冲突及干扰矩阵_filtered.xlsx'
    confusion_file = '混淆矩阵_filtered.xlsx'

    # Load data
    data_loader = DataLoader(cell_info_file, conflict_file, confusion_file)
    df1, df2, df3 = data_loader.load_data()

    # Construct matrices
    matrix_constructor = MatrixConstructor(df1, df2, df3)
    A, B, C = matrix_constructor.construct_matrices()

    # Create FitnessCalculator instance
    fitness_calculator = FitnessCalculator(A, B, C)

    # Initialize and run the evolutionary algorithm
    evolutionary_algo = EvolutionaryAlgorithm(fitness_calculator, num_cells = 2890, num_pcis = 1008, 
                 population_size=3, generations=100, mutation_rate=0.1,
                 crossover_rate=0.7, tournament_size=2)
    
    best_pci_assignment, best_fitness = evolutionary_algo.run()

    # Calculate MR components for the best assignment
    MR_sum, collision_sum, confusion_sum, interference_sum = fitness_calculator.calculate_MR(best_pci_assignment)

    print("\nOptimal PCI Assignment Found:")
    print(f"Total MR is: {MR_sum}")
    print(f"MR for collision: {collision_sum}")
    print(f"MR for confusion: {confusion_sum}")
    print(f"MR for interference: {interference_sum}")

    # Optionally, save the best PCI assignment to a file
    np.savetxt('best_pci_assignment.csv', best_pci_assignment, delimiter=',', fmt='%d')

if __name__ == "__main__":
    main()