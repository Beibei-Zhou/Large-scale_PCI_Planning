import random
import copy
import csv
from pci_planner import DataLoader, MatrixConstructor, FitnessCalculator

class HillClimber:
    """
    Hill Climber class to optimize PCI assignments using a hill climbing algorithm.
    """

    def __init__(self, cell_info_file, conflict_inference_file, confusion_file, 
                 num_pcis=1008, num_cells=2890, max_iterations=1000, csv_filename='hill_climbing_results.csv'):
        """
        Initialize the HillClimber with necessary data and parameters.

        Parameters:
        - cell_info_file: Path to the cell information Excel file.
        - conflict_inference_file: Path to the conflict inference Excel file.
        - confusion_file: Path to the confusion Excel file.
        - num_pcis: Total number of PCIs available.
        - num_cells: Total number of cells.
        - max_iterations: Maximum number of iterations to perform.
        - csv_filename: Name of the CSV file to save iteration results.
        """
        self.cell_info_file = cell_info_file
        self.conflict_inference_file = conflict_inference_file
        self.confusion_file = confusion_file
        self.num_pcis = num_pcis
        self.num_cells = num_cells
        self.max_iterations = max_iterations
        self.csv_filename = csv_filename
        self.baseline_MR = 50492016
        self.baseline_MR_collision = 47838
        self.baseline_MR_confusion = 0
        self.baseline_MR_interference = 50444178

        # Load data using DataLoader
        self.data_loader = DataLoader(cell_info_file, conflict_inference_file, confusion_file)
        df1, df2, df3 = self.data_loader.load_data()

        # Construct matrices A, B, C using MatrixConstructor
        self.matrix_constructor = MatrixConstructor(df1, df2, df3, num_pcis, num_cells)
        self.A, self.B, self.C = self.matrix_constructor.construct_matrices()

        # Initialize FitnessCalculator
        self.fitness_calculator = FitnessCalculator(self.A, self.B, self.C, num_pcis)

        # Initialize PCI assignment from the current PCI values
        self.pci_assignment = self.initialize_assignment(df1)

        # Calculate initial fitness
        (self.best_MR_sum, self.best_collision_sum, 
         self.best_confusion_sum, self.best_interference_sum) = self.fitness_calculator.calculate_MR(self.pci_assignment)
        self.best_assignment = copy.deepcopy(self.pci_assignment)

        # Initialize CSV file with headers
        self.initialize_csv()

    def initialize_csv(self):
        """
        Initialize the CSV file by writing the header row.
        """
        with open(self.csv_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            header = [
                'Iteration',
                'MR_sum',
                'Collision_sum',
                'Confusion_sum',
                'Interference_sum',
                'MR_Optimization_Percent',
                'Collision_Optimization_Percent',
                'Confusion_Increase',
                'Interference_Optimization_Percent'
            ]
            writer.writerow(header)

    def log_to_csv(self, iteration):
        """
        Log the current best metrics to the CSV file.

        Parameters:
        - iteration: The current iteration number.
        """
        with open(self.csv_filename, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            row = [
                iteration,
                self.best_MR_sum,
                self.best_collision_sum,
                self.best_confusion_sum,
                self.best_interference_sum,
                (self.baseline_MR - self.best_MR_sum)/self.baseline_MR,
                (self.baseline_MR_collision - self.best_collision_sum)/self.baseline_MR_collision,
                (self.best_confusion_sum - self.baseline_MR_confusion),
                (self.baseline_MR_interference - self.best_interference_sum)/self.baseline_MR_interference
            ]
            writer.writerow(row)

    def initialize_assignment(self, df1):
        """
        Extract the initial PCI assignment from the cell information DataFrame.

        Parameters:
        - df1: DataFrame containing cell information.

        Returns:
        - List of initial PCI assignments.
        """
        pci_assignment = []
        for _, row in df1.iterrows():
            pci = int(row['现网PCI'])
            pci_assignment.append(pci)
        return pci_assignment

    def get_neighbor(self):
        """
        Generate a neighboring PCI assignment by randomly changing the PCI of one cell.

        Returns:
        - A new PCI assignment list representing the neighbor.
        """
        neighbor = copy.deepcopy(self.pci_assignment)
        cell_to_change = random.randint(0, self.num_cells - 1)
        current_pci = neighbor[cell_to_change]
        
        # Choose a new PCI different from the current one
        new_pci = random.randint(0, self.num_pcis - 1)
        while new_pci == current_pci:
            new_pci = random.randint(0, self.num_pcis - 1)
        neighbor[cell_to_change] = new_pci
        return neighbor

    def run(self):
        """
        Execute the hill climbing optimization process.

        Returns:
        - best_assignment: The optimized PCI assignment list.
        - best_MR_sum: The best (lowest) MR sum achieved.
        - best_collision_sum: The corresponding collision sum.
        - best_confusion_sum: The corresponding confusion sum.
        - best_interference_sum: The corresponding interference sum.
        """
        iterations = 0
        patience = 0
        while iterations < self.max_iterations:
            improve = False
            for _ in range(self.num_cells):
                neighbor = self.get_neighbor()
                MR_sum, collision_sum, confusion_sum, interference_sum = self.fitness_calculator.calculate_MR(neighbor)

                if MR_sum < self.best_MR_sum:
                    self.best_MR_sum = MR_sum
                    self.best_collision_sum = collision_sum
                    self.best_confusion_sum = confusion_sum
                    self.best_interference_sum = interference_sum
                    self.best_assignment = copy.deepcopy(neighbor)
                    self.pci_assignment = neighbor
                    improve = True
                    
                    # Log the improvement to CSV
                    self.log_to_csv(iterations)
                    
                    print(f"Iteration {iterations}:")
                    print(f"MR_sum:{self.best_MR_sum}")
                    print(f"MR_collision_sum:{self.best_collision_sum}")
                    print(f"MR_confusion_sum:{self.best_confusion_sum}")
                    print(f"MR_inference_sum:{self.best_interference_sum}")
                    print(f"MR_sum Optimization Percent:{(self.baseline_MR - self.best_MR_sum)/self.baseline_MR:.6f}")
                    print(f"MR Collision Optimization Percent:{(self.baseline_MR_collision - self.best_collision_sum)/self.baseline_MR_collision:.6f}")
                    print(f"MR Confusion MR Increasing:{(self.best_confusion_sum - self.baseline_MR_confusion)}")
                    print(f"MR Interference Optimization Percent:{(self.baseline_MR_interference - self.best_interference_sum)/self.baseline_MR_interference:.6f}")
                    print("--------------------------------------------------------------------------------")
                    break
            if not improve:
                # Log the non-improvement to CSV
                self.log_to_csv(iterations)
                
                print(f"Iteration {iterations}:")
                print(f"MR_sum:{self.best_MR_sum}")
                print(f"MR_collision_sum:{self.best_collision_sum}")
                print(f"MR_confusion_sum:{self.best_confusion_sum}")
                print(f"MR_inference_sum:{self.best_interference_sum}")
                print(f"MR_sum Optimization Percent:{(self.baseline_MR - self.best_MR_sum)/self.baseline_MR:.6f}")
                print(f"MR Collision Optimization Percent:{(self.baseline_MR_collision - self.best_collision_sum)/self.baseline_MR_collision:.6f}")
                print(f"MR Confusion MR Increasing:{(self.best_confusion_sum - self.baseline_MR_confusion)}")
                print(f"MR Interference Optimization Percent:{(self.baseline_MR_interference - self.best_interference_sum)/self.baseline_MR_interference:.6f}")
                print("--------------------------------------------------------------------------------")    
                patience += 1

            if patience >= 2:
                print("****************************************")
                print(f"The optimization has reached convergence at iteration {iterations}")
                print("****************************************")
                break

            iterations += 1

        print("Hill Climbing completed.")
        print(f"Best MR_sum: {self.best_MR_sum}")
        print(f"Collision: {self.best_collision_sum}")
        print(f"Confusion: {self.best_confusion_sum}")
        print(f"Interference: {self.best_interference_sum}")

        return (self.best_assignment, self.best_MR_sum, 
                self.best_collision_sum, self.best_confusion_sum, 
                self.best_interference_sum)

