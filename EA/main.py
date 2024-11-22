# main.py
import numpy as np
import random
from pci_planner import DataLoader, MatrixConstructor, FitnessCalculator
from evo_alg import EvolutionaryAlgorithm
import time

def main():
    # Set random seeds for reproducibility
    # random.seed(42)
    # np.random.seed(42)

    # File paths
    cell_info_file = '小区信息_filtered.xlsx'
    conflict_file = '冲突及干扰矩阵_filtered.xlsx'
    confusion_file = '混淆矩阵_filtered.xlsx'

    # Load data
    data_loader = DataLoader(cell_info_file, conflict_file, confusion_file)
    df1, df2, df3 = data_loader.load_data()

    # Extract the baseline PCI assignemnt from df1
    baseline_pci_assignment = df1['现网PCI'].astype(int).tolist()

    # Construct matrices
    matrix_constructor = MatrixConstructor(df1, df2, df3)
    A, B, C = matrix_constructor.construct_matrices()

    # Create FitnessCalculator instance
    fitness_calculator = FitnessCalculator(A, B, C)

    # Initialize and run the evolutionary algorithm
    start_time = time.time()
    evolutionary_algo = EvolutionaryAlgorithm(
        fitness_calculator, 
        num_cells = 2890, 
        num_pcis = 1008, 
        population_size=100, 
        generations=2000, 
        mutation_rate=0.5,
        crossover_rate=0.7, 
        tournament_size=5,
        optimized_population = False,
        baseline_pci_assignment=baseline_pci_assignment,
        hc_iterations=2500,
        selection_method = 'roulette')
    end_time = time.time()
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
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()