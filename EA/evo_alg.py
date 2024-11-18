import numpy as np
import random
import copy
import csv

class EvolutionaryAlgorithm:
    """Class implementing the Evolutionary Algorithm for PCI planning"""
    def __init__(self, fitness_calculator, num_cells=2890, num_pcis=1008, 
                 population_size=100, generations=100, mutation_rate=0.1,
                 crossover_rate=0.7, tournament_size=5):
        self.fitness_calculator = fitness_calculator
        self.num_cells = num_cells
        self.num_pcis = num_pcis
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

    def initialize_population(self):
        """Initialize population with random PCI assignments."""
        population = [np.random.randint(0, self.num_pcis, size=self.num_cells)
                      for _ in range(self.population_size)]
        return population

    def evaluate_fitness(self, population):
        """Evaluate fitness of each individual in the population."""
        fitness_scores = []
        for individual in population:
            MR_sum, _, _, _ = self.fitness_calculator.calculate_MR(individual)
            fitness_scores.append(MR_sum)
        return fitness_scores
    
    def select_individuals(self, population, fitness_scores):
        """Select individuals using tournament selection"""
        selected_individuals = []
        for _ in range(self.population_size):
            tournament = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
            tournament_winner = min(tournament, key=lambda x: x[1])
            selected_individuals.append(copy.deepcopy(tournament_winner[0]))
        return selected_individuals
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.num_cells - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        else:
            # If no crossover, children are copies of parents
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
        return child1, child2
    
    def mutate(self, individual):
        """Mutate an individual"""
        for i in range(self.num_cells):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, self.num_pcis - 1)
        return individual

    def run(self):
        """Execute the evolutionary algorithm."""
        population = self.initialize_population()
        
        # Initialize best overall fitness and MR components
        best_overall_fitness = float('inf')
        best_overall_individual = None
        best_overall_collision = None
        best_overall_confusion = None
        best_overall_interference = None

        # CSV headers
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

            # Write best MR_sum till now to CSV
            with open('best_overall.csv', mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([generation + 1, best_overall_fitness, best_overall_collision, best_overall_confusion, best_overall_interference])

            # Selection
            selected_individuals = self.select_individuals(population, fitness_scores)

            # Generate a new population
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = random.choice(selected_individuals)
                parent2 = random.choice(selected_individuals)

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # Replace the old population with the new one
            population = new_population[:self.population_size]

        return best_overall_individual, best_overall_fitness
