import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Tuple
import time

def data_preprocessing(path= './fri26_d.txt')->np.array:
    with open(path) as file:
        data_str = file.read().split()
        data = list(map(int,data_str)) # transfer type "str" to "int"
    dim = int(len(data)**0.5)
    dis_matrix = np.array(data).reshape(dim,dim) # reshape to symmetric matrix
    return dis_matrix

class TSP(): 
    def __init__(self,population_size:int,crossover_rate:float,mutation_rate:float,iteration:int,data:np.ndarray):
        self.pop_size = population_size
        self.cross_rate = crossover_rate
        self.mutate_rate = mutation_rate
        self.N_GENERATIONS = iteration
        self.dis_arr = data
        self.dim = data.shape[0]
        self.population = np.vstack([(random.sample(range(self.dim),self.dim)) for _ in range(self.pop_size)])
        
    def fitness_calculate(self) -> Tuple[list,int,np.ndarray]:
        first_elem = self.population[:, 0] # first element
        pop = np.hstack((self.population, first_elem.reshape(-1,1))) # combines array horizontally
        fitness = []
        for chrom in pop:
            # calculate the distance of each route 
            route = 0
            for idx in range(0,self.dim):
                route += self.dis_arr[chrom[idx]][chrom[idx+1]]
            fitness.append(route)       
        # minimize distance and route
        min_fitness = min(fitness)
        min_fitness_idx = fitness.index(min_fitness)
        min_route = pop[min_fitness_idx]
        return fitness,min_fitness,min_route
    
    # Roulette wheel
    def selection_RouletteWheel(self,fitness:list)->np.ndarray:
        fit_invers = [1/f for f in fitness] # The minimization problem must be inverse
        prob = fit_invers/sum(fit_invers) # Roulette wheel probability
        idx = np.random.choice(np.arange(self.pop_size), size=int(self.pop_size/2), replace=True, p=prob)
        return self.population[idx]
    # Elitism
    def selection_Elitism(self,fitness:list)->np.ndarray:
        idx = [fitness.index(value) for value in sorted(fitness)[:int(self.pop_size/2)]]
        return self.population[idx]
    
    def crossover_mutation(self,parent:np.ndarray)->np.ndarray:
        parent_len = int(self.pop_size/2)
        offspring = np.zeros((parent_len,self.dim)) # create offspring
        for i in range(parent_len):
            prob_gens = random.uniform(0,1) # random rate
            # crossover:one point
            if prob_gens < self.cross_rate:
                cross_chrom = parent[random.randint(0,parent.shape[0]-1)] # Randomly select a chromosome
                cross_point = random.randint(1,self.dim-1) # crossover point
                cross_gens = cross_chrom[cross_point:] # crossover element

                del_idx = [int(np.where(parent[i]==elem)[0]) for elem in cross_gens] # Index of the current chromosome to be deleted
                cross_del = np.delete(parent[i],del_idx) # delete element
                inser_point = random.randint(0,len(cross_del)-1) # Select the index to be inserted
                offspring[i] = np.concatenate((cross_del[:inser_point],cross_gens,cross_del[inser_point:])) # Merge and join the offspring

            # mutation:two point(Route cannot be duplicated)
            elif prob_gens < self.mutate_rate:
                first_point = random.randint(0,int((self.dim-1)/2)) # first mutation point
                second_point = random.randint(first_point,self.dim-1) # second mutation point

                offspring[i] = parent[i]
                offspring[i][first_point] = parent[i][second_point]
                offspring[i][second_point] = parent[i][first_point]
            else:
                offspring[i] = parent[i]
        return offspring
    
    def iteration_run(self)->Tuple[list,list]:
        self.history_fitness = []
        self.history_path = []
        for _ in range(self.N_GENERATIONS):
            fitness,min_fitness,min_route = self.fitness_calculate()
            self.history_fitness.append(min_fitness)
            self.history_path.append(min_route)
                
            parent = self.selection_Elitism(fitness)
            offspring = self.crossover_mutation(parent)
            self.population[:] = np.concatenate((parent,offspring)) # New generation of chromosomes
        return self.history_fitness,self.history_path


# parameter
population_size = 200
crossover_rate = 0.9
mutation_rate = 0.2
iteration = 200
data = data_preprocessing()

# Start Iteration
start_time = time.time()
GA = TSP(population_size,crossover_rate,mutation_rate,iteration,data)
GA.iteration_run()
fitness_list = GA.history_fitness
path_list = GA.history_path
best_fitness = min(fitness_list)
best_path = path_list[fitness_list.index(best_fitness)]
end_time = time.time()

print("Runing time:%.2f seconds" %(end_time-start_time))
print(f"Shortest distance: {best_fitness} \nShortest route:{best_path}")

plt.grid(axis='y')
plt.title('The convergence history',fontsize=20)
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.plot(fitness_list)
plt.plot(fitness_list.index(best_fitness),best_fitness,'ro')
plt.annotate(best_fitness,xy=(fitness_list.index(best_fitness),best_fitness),xytext=(fitness_list.index(best_fitness),best_fitness+50))
plt.show()

# Experimentally conducted 10 times GA to observe the convergence
exp_num = 10
history_best_fitness = []
history_best_path = []

history_best_fitness.append(best_fitness)
history_best_path.append(best_path)
for _ in range(exp_num-1):
    GA = TSP(population_size,crossover_rate,mutation_rate,iteration,data)
    GA.iteration_run()
    fitness_list = GA.history_fitness
    path_list = GA.history_path
    best_fitness = min(fitness_list)
    best_path = path_list[fitness_list.index(best_fitness)]
    
    history_best_fitness.append(best_fitness)
    history_best_path.append(best_path)

exp_best_fitness = min(history_best_fitness)
exp_best_route = history_best_path[history_best_fitness.index(exp_best_fitness)]
print(f"Shortest distance in {exp_num} experiments:{exp_best_fitness}")
print(f"Shortest route in {exp_num} experiments:{exp_best_route}")
plt.grid(axis='y')
plt.title(f"Execute {exp_num} times",fontsize=20)
plt.xlabel("Number of executions")
plt.ylabel("Fitness")
plt.plot(history_best_fitness,'-o')
for i in range(len(history_best_fitness)):
    if history_best_fitness[i] == min(history_best_fitness) or history_best_fitness[i] == max(history_best_fitness):
        plt.text(i-.75,history_best_fitness[i]+1,history_best_fitness[i])
plt.show()