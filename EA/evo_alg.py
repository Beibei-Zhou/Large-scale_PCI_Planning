import numpy as np
import random
import copy
import csv

class EvolutionaryAlgorithm:
    """Class implementing the Evolutionary Algorithm for PCI planning"""
    def __init__(self, fitness_calculator, num_cells=2890, num_pcis=1008, 
                 population_size=100, generations=100, mutation_rate=0.1,
                 crossover_rate=0.7, tournament_size=5, optimized_population=False, 
                 baseline_pci_assignment=None, hc_iterations=1000, selection_method = 'tournament'):
        
        """
        Initialize the Evolutionary Algorithm with necessary parameters.

        Parameters:
        - fitness_calculator: Instance of FitnessCalculator to evaluate individuals.
        - num_cells: Total number of cells.
        - num_pcis: Total number of PCIs available.
        - population_size: Number of individuals in the population.
        - generations: Number of generations to evolve.
        - mutation_rate: Probability of mutating a gene.
        - crossover_rate: Probability of performing crossover.
        - tournament_size: Number of individuals participating in tournament selection.
        - optimized_population: Flag to determine if initial population should be optimized.
        - baseline_pci_assignment: Baseline PCI assignment list.
        - hc_iterations: Maximum hill-climbing iterations per individual.
        """
        self.fitness_calculator = fitness_calculator
        self.num_cells = num_cells
        self.num_pcis = num_pcis
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.optimized_population = optimized_population
        self.baseline_pci_assignment = baseline_pci_assignment
        self.hc_iterations = hc_iterations
        self.baseline_MR = 50492016
        self.baseline_MR_collision = 47838
        self.baseline_MR_confusion = 0
        self.baseline_MR_interference = 50444178
        self.selection_method = selection_method.lower()
        self.valid_selection_methods = ['tournament', 'roulette']
        
        if self.selection_method not in self.valid_selection_methods:
            raise ValueError(f"Invalid selection_method '{selection_method}'. "
                             f"Choose from {self.valid_selection_methods}.")
        
        if self.optimized_population:
            if self.baseline_pci_assignment is None:
                raise ValueError("Baseline PCI assignment must be provided when optimized_population is True.")
            if len(self.baseline_pci_assignment) != self.num_cells:
                raise ValueError("Baseline PCI assignment length does not match num_cells.")
            # Convert baseline_pci_assignment to NumPy array if it's not
            if not isinstance(self.baseline_pci_assignment, np.ndarray):
                self.baseline_pci_assignment = np.array(self.baseline_pci_assignment)
        
        # Initialize best fitness based on baseline if optimized_population is True
        if self.optimized_population and self.baseline_pci_assignment is not None:
            self.best_MR_sum, self.best_collision_sum, self.best_confusion_sum, self.best_interference_sum = \
                self.fitness_calculator.calculate_MR(self.baseline_pci_assignment)
            self.best_assignment = copy.deepcopy(self.baseline_pci_assignment)
        else:
            # If not using optimized_population, initialize best fitness to infinity
            self.best_MR_sum = float('inf')
            self.best_collision_sum = None
            self.best_confusion_sum = None
            self.best_interference_sum = None
            self.best_assignment = None
                
    def initialize_population(self):
        """Initialize population with either optimized or random PCI assignments."""
        population = []
        if self.optimized_population:
            print("*******************************************************")
            print("Initializing a population using hill climber optimization")
            print("*******************************************************")
            for i in range(self.population_size):
                print(f"Optimizing Individual No.{i+1}/{self.population_size}")
                # Start with a NumPy array copy of the baseline
                individual = copy.deepcopy(self.baseline_pci_assignment)
                current_best_MR_sum = self.best_MR_sum
                current_best_assignment = copy.deepcopy(individual)
                patience = 0
                
                for iter in range(self.hc_iterations):
                    improve = False
                    for _ in range(self.num_cells):
                        # Generate a neighbor
                        neighbor = self.get_neighbor(current_best_assignment)
                        MR_sum, collision_sum, confusion_sum, interference_sum = self.fitness_calculator.calculate_MR(neighbor)
                        
                        if MR_sum < current_best_MR_sum:
                            current_best_MR_sum = MR_sum
                            current_best_assignment = copy.deepcopy(neighbor)
                            improve = True
                            print(f"Individual No.{i+1}, Iteration {iter+1}: Improved MR_sum to {MR_sum}")
                            break  # Exit the inner loop as improvement has been found
                    
                    if not improve:
                        # If no improvement, consider stopping early
                        print(f"Individual No.{i+1}, Iteration {iter+1}: No improvement")
                        patience += 1
                
                    if patience >= 2:
                        print("****************************************")
                        print(f"The optimization has reached convergence at iteration {iter+1} for individual No. {i+1}")
                        print("****************************************")
                        break                
                # After hill climbing, append the optimized individual to the population
                population.append(copy.deepcopy(current_best_assignment))
                print(f"Finished optimizing Individual No.{i+1}: MR_sum = {current_best_MR_sum}")
                print("--------------------------------------------------------------------------------")

            print("*******************************************************")
            print("Population initialization using hill climber completed")
            print("*******************************************************")
        else:       
            print("*******************************************************")
            print("Initializing a population randomly")
            print("*******************************************************")     
            population = [np.random.randint(0, self.num_pcis, size=self.num_cells)
                          for _ in range(self.population_size)]
            print("Population initialized randomly.")
        return population

    def get_neighbor(self, individual):
        """Generate a neighbor PCI assignment by mutating a random cell."""
        neighbor = copy.deepcopy(individual)
        cell_to_change = random.randint(0, self.num_cells - 1)  # Corrected upper bound
        current_pci = neighbor[cell_to_change]
        new_pci = random.randint(0, self.num_pcis - 1)
        while new_pci == current_pci:
            new_pci = random.randint(0, self.num_pcis - 1)
        neighbor[cell_to_change] = new_pci
        return neighbor

    def evaluate_fitness(self, population):
        """Evaluate fitness of each individual in the population."""
        fitness_scores = []
        for individual in population:
            MR_sum, _, _, _ = self.fitness_calculator.calculate_MR(individual)
            fitness_scores.append(MR_sum)
        return fitness_scores
    
    def select_individuals_tournament(self, population, fitness_scores):
        """Select individuals using tournament selection."""
        selected_individuals = []
        for _ in range(self.population_size):
            tournament = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
            tournament_winner = min(tournament, key=lambda x: x[1])
            # Append both individual and its fitness
            selected_individuals.append((copy.deepcopy(tournament_winner[0]), tournament_winner[1]))
        return selected_individuals
    
    def select_individuals_roulette(self, population, fitness_scores):
        """Select individuals using roulette selection"""
        selected_individuals = []

        # Ensure there are individuals to select from
        if not population or not fitness_scores:
            return selected_individuals

        # Transform fitness scores: since lower fitness is better, invert the scores
        epsilon = 1e-6
        try:
            inverted_fitness = [1.0 / (f + epsilon) for f in fitness_scores]
        except ZeroDivisionError:
            raise ValueError("Fitness scores must be non-negative")
        
        total_fitness = sum(inverted_fitness)

        if total_fitness == 0:
            raise ValueError("Total inverted fitness is zero. Check fitness scores")
        
        selection_probs = [f / total_fitness for f in inverted_fitness]
        
        # Create cumulative probability distribution
        cumulative_probs = []
        cumulative = 0.0

        for prob in selection_probs:
            cumulative += prob
            cumulative_probs.append(cumulative)
        
        # Selection process:
        for _ in range(self.population_size):
            r = random.random()
            # Find the individual corresponding to the ramdom number
            for index, cumulative_prob in enumerate(cumulative_probs):
                if r <= cumulative_prob:
                    selected_individual = copy.deepcopy(population[index])
                    selected_fitness = fitness_scores[index]
                    selected_individuals.append((selected_individual, selected_fitness))
                    break
        return selected_individuals
    
    def crossover(self, parent1, parent2, parent1_fitness, parent2_fitness):
        """
        Perform crossover between two parents and ensure that the combined
        fitness of the children is better than that of the parents.
        
        Parameters:
        - parent1: First parent individual (numpy array).
        - parent2: Second parent individual (numpy array).
        - parent1_fitness: Fitness score of parent1.
        - parent2_fitness: Fitness score of parent2.
        
        Returns:
        - child1: First child individual.
        - child2: Second child individual.
        """
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.num_cells - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            
            # Calculate fitness for children
            child1_fitness = self.fitness_calculator.calculate_MR(child1)[0]
            child2_fitness = self.fitness_calculator.calculate_MR(child2)[0]
            
            # Compare combined fitness
            parents_fitness = parent1_fitness + parent2_fitness
            children_fitness = child1_fitness + child2_fitness
            
            if children_fitness < parents_fitness:
                return child1, child2
            else:
                # If children are not better, retain parents
                return parent1.copy(), parent2.copy()
        else:
            # If no crossover, children are copies of parents
            return parent1.copy(), parent2.copy()
    
    def mutate_draft(self, individual):
        """Mutate an individual."""
        for i in range(self.num_cells):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, self.num_pcis - 1)
        return individual
    
    def get_neighbor(self, individual):
        """
        Generate a neighboring PCI assignment by randomly changing the PCI of one cell.

        Parameters:
        - individual (numpy array): The current PCI assignment array.

        Returns:
        - neighbor (numpy array): A new PCI assignment array representing the neighbor.
        """
        neighbor = copy.deepcopy(individual)
        cell_to_change = random.randint(0, self.num_cells - 1)
        current_pci = neighbor[cell_to_change]
        
        # Choose a new PCI different from the current one
        new_pci = random.randint(0, self.num_pcis - 1)
        while new_pci == current_pci:
            new_pci = random.randint(0, self.num_pcis - 1)
        neighbor[cell_to_change] = new_pci
        return neighbor

    def mutate(self, individual):
        """
        Mutate an individual using a hill-climbing approach.

        For the given individual, iteratively search for better neighboring PCI assignments
        by altering one gene at a time. Update the individual if an improvement is found.
        The process continues until no improvement is possible or a maximum number of
        hill-climbing iterations is reached.

        Parameters:
        - individual (numpy array): The PCI assignment array of the individual.

        Returns:
        - individual (numpy array): The mutated (and possibly optimized) individual.
        """
        max_hc_iterations = self.hc_iterations  # Define a maximum number of hill-climbing iterations per mutation
        hc_iterations = 0
        patience = 0
        # Calculate the original fitness of the individual
        original_fitness, _, _, _ = self.fitness_calculator.calculate_MR(individual)

        while hc_iterations < max_hc_iterations:
            improvement = False
            hc_iterations += 1
            for _ in range(self.num_cells):
                # Generate a neighbor
                neighbor = self.get_neighbor(individual)
                neighbor_fitness, _, _, _ = self.fitness_calculator.calculate_MR(neighbor)

                # Check if the neighbor is better
                if neighbor_fitness < original_fitness:
                    # Update the individual to the better neighbor
                    individual = copy.deepcopy(neighbor)
                    original_fitness = neighbor_fitness
                    improvement = True

                    # Optional: Log the improvement (you can customize this as needed)
                    #print(f"Hill-Climbing Iteration {hc_iterations}: Improved MR_sum to {neighbor_fitness}")
                    break
            if not improvement:
                patience += 1
            if patience >= 2:
                break
        return individual
    
    def write_population_to_csv(self, population, generation):
        """
        Write the entire population's PCI assignments to a CSV file for the given generation.
        
        Parameters:
        - population (list of numpy arrays): The current population.
        - generation (int): The current generation number.
        """
        # Define the filename
        population_filename = f'population_gen_{generation}.csv'

        # Open the file
        with open(population_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            
            # Write the header
            header = ['Individual_ID'] + [f'Cell_{i+1}' for i in range(self.num_cells)]
            writer.writerow(header)
            
            # Write each individual
            for idx, individual in enumerate(population):
                # Ensure individual is a NumPy array
                if isinstance(individual, np.ndarray):
                    row = [idx + 1] + individual.tolist()
                else:
                    # If individual is a list, proceed without calling tolist()
                    row = [idx + 1] + individual
                writer.writerow(row)
    
    def run(self):
        """Execute the evolutionary algorithm."""
        population = self.initialize_population()
        
        # Initialize best overall fitness and MR components
        if self.optimized_population and self.best_assignment is not None:
            best_overall_fitness = self.best_MR_sum
            best_overall_individual = copy.deepcopy(self.best_assignment)
            best_overall_collision = self.best_collision_sum
            best_overall_confusion = self.best_confusion_sum
            best_overall_interference = self.best_interference_sum
        else:
            best_overall_fitness = float('inf')
            best_overall_individual = None
            best_overall_collision = None
            best_overall_confusion = None
            best_overall_interference = None

        # CSV headers for best individuals
        with open('best_in_generation.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'MR_sum', 'Collision', 'Confusion', 'Interference'])

        with open('best_overall.csv', mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'Best_MR_sum_till_now', 'Collision', 'Confusion', 'Interference'])

        for generation in range(self.generations):
            print(f"\nGeneration {generation + 1}")

            # Evaluate fitness
            fitness_scores = self.evaluate_fitness(population)

            # Find the best individual in current generation
            best_fitness = min(fitness_scores)
            best_index = np.argmin(fitness_scores)
            best_individual = population[best_index]

            # Get MR components for the best individual
            MR_sum, collision_sum, confusion_sum, interference_sum = self.fitness_calculator.calculate_MR(best_individual)

            # Print best MR_sum and MR components in current generation
            print(f"Best MR_sum in current generation: {MR_sum}")
            print(f"MR collision: {collision_sum}, MR confusion: {confusion_sum}, MR interference: {interference_sum}")

            # Write best MR_sum in current generation to CSV
            with open('best_in_generation.csv', mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([generation + 1, MR_sum, collision_sum, confusion_sum, interference_sum])

            # Update best overall fitness and MR components if current best is better
            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_individual = copy.deepcopy(best_individual)
                best_overall_collision = collision_sum
                best_overall_confusion = confusion_sum
                best_overall_interference = interference_sum

            # Print best MR_sum and MR components from start till now
            print(f"Best MR_sum from start till now: {best_overall_fitness}")
            print(f"MR collision: {best_overall_collision}, MR confusion: {best_overall_confusion}, MR interference: {best_overall_interference}")
            print(f"MR_sum Optimization Percent: {(self.baseline_MR - best_overall_fitness)/self.baseline_MR:.6f}")
            print(f"MR Collision Optimization Percent: {(self.baseline_MR_collision - best_overall_collision)/self.baseline_MR_collision:.6f}")
            print(f"MR Confusion MR Increasing: {(best_overall_confusion - self.baseline_MR_confusion)}")
            print(f"MR Interference Optimization Percent: {(self.baseline_MR_interference - best_overall_interference)/self.baseline_MR_interference:.6f}")
            
            # Write best MR_sum till now to CSV
            with open('best_overall.csv', mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([generation + 1, best_overall_fitness, best_overall_collision, best_overall_confusion, best_overall_interference])

            # **Write Entire Population to CSV**
            self.write_population_to_csv(population, generation + 1)

            # Selection
            if self.selection_method == 'tournament':
                selected_individuals = self.select_individuals_tournament(population, fitness_scores)
            elif self.selection_method == 'roulette':
                selected_individuals = self.select_individuals_roulette(population, fitness_scores)
            else:
                raise ValueError(f"Unsupported selection method")

            # Generate a new population
            new_population = []
            while len(new_population) < self.population_size:
                # Randomly select two parents
                parent1, parent1_fitness = random.choice(selected_individuals)
                parent2, parent2_fitness = random.choice(selected_individuals)

                # Crossover with fitness comparison
                child1, child2 = self.crossover(parent1, parent2, parent1_fitness, parent2_fitness)

                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # Replace the old population with the new one
            population = new_population[:self.population_size]

        return best_overall_individual, best_overall_fitness
