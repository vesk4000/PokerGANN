import pygad
import numpy
import numba
from tensorflow import keras
import pygad.kerasga
import PokerGame
import os
from os.path import exists
from matplotlib import pyplot
import psutil
import time


def seconds_elapsed():
	p = psutil.Process(os.getpid())
	return time.time() - p.create_time()

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


num_generations = 300
num_parents_mating = 5
initial_population = keras_ga.population_weights

# x = numpy.array([0, 1, 2, 3, 4, 5])
# y = numpy.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
# x = numpy.linspace(0, 5)
# y = numpy.sin(x)
x = numpy.linspace(0, num_generations, num=(num_generations + 1))
# y1 = []
# for i in range(0, num_generations + 1):
# 	y1.append(0.5)
y = numpy.linspace(0, 0.5, num=(num_generations + 1))


pyplot.ion()
fig = pyplot.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')

global xx
xx = []
global yy
yy = []
global curr_gen
curr_gen = 1


def callback_generation(ga_instance):
	global xx
	global yy
	global curr_gen
	print("Generation = {generation}".format(generation=ga_instance.generations_completed))
	print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
	xx.append(curr_gen)
	yy.append(ga_instance.best_solution()[1])
	line1.set_xdata(numpy.array(xx))
	line1.set_ydata(numpy.array(yy))
	# ax.autoscale()
	# ax.set_autoscale_on(True)
	# ax.autoscale_view()
	fig.canvas.draw()
	fig.canvas.flush_events()
	curr_gen += 1
	if seconds_elapsed() > 10 * 60:
		return "stop"



# if not exists(os.getcwd() + "\\model.pkl"):
ga_instance = pygad.GA(num_generations=num_generations, 
	num_parents_mating=num_parents_mating, 
	initial_population=initial_population,
	fitness_func=fitness_func,
	on_generation=callback_generation,
	mutation_probability=0.2,
	save_solutions = True
)
# else:
# 	ga_instance = pygad.load(filename = ("model"))

ga_instance.run()

pyplot.ioff()

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

ga_instance.save(filename = "model")