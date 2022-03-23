import pygad
import numpy
import numba
from tensorflow import keras
import pygad.kerasga
import PokerGame
import os
from os.path import exists

input_layer = keras.layers.Input(3)
para_relu = keras.layers.PReLU()
hidden_layer1 = keras.layers.Dense(2, activation = "elu") # relu
hidden_layer2 = keras.layers.Dense(2, activation = "elu")
output_layer = keras.layers.Dense(1, activation = "sigmoid")

model = keras.Sequential()
model.add(input_layer)
model.add(hidden_layer1)
model.add(hidden_layer2)
model.add(output_layer)

keras_ga = pygad.kerasga.KerasGA(model = model, num_solutions = 10)

print("Hello there " + pygad.__version__)

def fitness_func(solution, sol_idx):
	global keras_ga, model
	model_weights_matrix \
		= pygad.kerasga.model_weights_as_matrix(model = model, weights_vector = solution)
	model.set_weights(weights = model_weights_matrix)
	strat = PokerGame.PokerStrategy(model, 3, 1)
	#return 1.0 / (strat.compute_average_winnings())
	return strat.compute_expected_profit()

num_generations = 10
num_parents_mating = 5
initial_population = keras_ga.population_weights

def callback_generation(ga_instance):
	print("Generation = {generation}".format(generation=ga_instance.generations_completed))
	print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

ga_instance = pygad.GA(num_generations=num_generations, 
	num_parents_mating=num_parents_mating, 
	initial_population=initial_population,
	fitness_func=fitness_func,
	on_generation=callback_generation,
	mutation_probability=0.2,
	save_solutions = True
)

if not exists(os.getcwd() + "\\model.pkl"):
	print("not exists")
else:
	ga_instance = pygad.load(filename = ("model"))

ga_instance.run()

ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

ga_instance.plot_genes(title = "Iteration vs. Genes", linewidth = 2);

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
model.set_weights(best_solution_weights)
strat = PokerGame.PokerStrategy(model, 3, 1)
print(strat)

ga_instance.save(filename = ("model"))