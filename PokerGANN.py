from pdb import find_function
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
from matplotlib.animation import FuncAnimation
import _thread
import threading


global ax
plot_x_data = []
plot_y_data = []
plot_data_lock = threading.RLock()
global keras_ga, model, ani

def main():
	train([[3, 1], [5, 3, 2], [31, 4, 5]], num_generations = 30, save = "model")


def train(tests : list[list[int]], load : str = None, save : str = None, time_limit : int = None, num_generations : int = None, mutation_probability : int = 0.2):
	ga_instance, model = init(load, time_limit, num_generations)
	def fitness_func(solution, sol_idx):
		return evaluate_solution(solution, model, tests)
	ga_instance.fitness_func = fitness_func

	_thread.start_new_thread(runPlotAnimation, ())

	ga_instance.run()

	if save != None:
		ga_instance.save(save)

def test():
	pass


# Note: Changing anything here will likely invalidate any models that have been saved
def init(load : str = None, time_limit : int = None, num_generations : int = None, mutation_probability : int = 0.2) -> tuple[pygad.GA, keras.Sequential]:
	NUM_SOLUTIONS = 10
	NUM_PARENTS_MATING = 5

	input_layer = keras.layers.Input(3)
	hidden_layer1 = keras.layers.Dense(2, activation = "elu")
	hidden_layer2 = keras.layers.Dense(2, activation = "elu")
	output_layer = keras.layers.Dense(1, activation = "sigmoid")

	model = keras.Sequential()
	model.add(input_layer)
	model.add(hidden_layer1)
	model.add(hidden_layer2)
	model.add(output_layer)

	keras_ga = pygad.kerasga.KerasGA(model = model, num_solutions = NUM_SOLUTIONS)
	initial_population = keras_ga.population_weights

	def fitness_func(solution, sol_idx):
		pass

	if load == None:
		ga_instance = pygad.GA(
			num_generations = 0,
			fitness_func = fitness_func,
			num_parents_mating = NUM_PARENTS_MATING,
			initial_population = initial_population,
			mutation_probability = mutation_probability,
			save_solutions = True
		)
		if num_generations == None:
			num_generations = 10
	else:
		ga_instance = load(load)
	
	if num_generations != None:
		ga_instance.num_generations = num_generations

	if time_limit == None:
		on_generation_func = callback_generation
	else:
		def time_limit_func(_ga_instance):
			callback_generation(_ga_instance)
			if seconds_elapsed() > time_limit:
				return "stop"
		on_generation_func = time_limit_func
	ga_instance.on_generation = on_generation_func

	def fitness_func(solution, sol_idx):
		return evaluate_solution()
	
	return ga_instance, model


def runPlotAnimation():
	global ani
	fig = pyplot.figure()
	global ax
	ax = fig.add_subplot(111)
	ax.plot(plot_x_data, plot_y_data, 'b-')
	ani = FuncAnimation(fig, plotAnimate, interval = 1000)
	pyplot.tight_layout()
	pyplot.show()


def plotAnimate(i):
	plot_data_lock.acquire()
	global ax
	ax.cla()
	ax.plot(plot_x_data, plot_y_data, 'b-')
	plot_data_lock.release()


def seconds_elapsed():
	p = psutil.Process(os.getpid())
	return time.time() - p.create_time()


def evaluate_solution(solution, model, tests : list[list[int]]):
	profits = []
	for test in tests:
		if len(test) == 2:
			num_cards = test[0]
			bet_to_ante_ratio = test[1]
		elif len(test) == 3:
			num_cards = test[0]
			bet_to_ante_ratio = test[2] / test[1]
		model.set_weights(weights = pygad.kerasga.model_weights_as_matrix(model = model, weights_vector = solution))
		#strat = PokerGame.PokerStrategy(model, num_cards, bet_to_ante_ratio)
		strat = PokerGame.newPokerStrategy(model, num_cards, bet_to_ante_ratio)
		#profits.append(strat.compute_expected_profit())
		profits.append(PokerGame.compute_expected_profit(strat))
	return numpy.average(profits)


def callback_generation(ga_instance):
	global plot_x_data
	global plot_y_data
	global plot_data_lock
	print("Generation = {generation}".format(generation=ga_instance.generations_completed))
	print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
	plot_data_lock.acquire()
	plot_x_data = numpy.linspace(1, len(ga_instance.best_solutions_fitness), num = len(ga_instance.best_solutions_fitness) + 1)
	plot_y_data = ga_instance.best_solutions_fitness
	plot_data_lock.release()

# ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# ga_instance.plot_genes(title = "Iteration vs. Genes", linewidth = 2);

# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
# print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# # Fetch the parameters of the best solution.
# best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
# model.set_weights(best_solution_weights)
# strat = PokerGame.PokerStrategy(model, 3, 1)
# print(strat)

if __name__ == "__main__":
	main()