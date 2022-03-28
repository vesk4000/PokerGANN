from pdb import find_function
from pyexpat import model
import warnings
from matplotlib import animation
import matplotlib
import pygad
import numpy
import numba
import pygad.kerasga
import PokerGame
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
matplotlib.set_loglevel("error")
warnings.filterwarnings("ignore")
from tensorflow import keras
from os.path import exists
from matplotlib import pyplot
import psutil
import time
from matplotlib.animation import FuncAnimation
import _thread
import threading
import math
# import pyximport
# pyximport.install()
# import PokerGameCython

numpy.seterr(divide = "print")


def main():
	# [3, 1], [5, 3, 2], [31, 4, 5]
	train([[5, 0.66]], num_generations = 2000, save = "model")


global_tests : list[list[int]] = None
global_model : keras.Sequential = None
global_time_limit : int = None
plot_x_data = []
plot_y_data = []
plot_data_lock = threading.RLock()
ax = None
keras_ga = None
animation_thread = None

ani_thread = None
def train(
	tests : list[list[int]],
	load : str = None,
	save : str = None,
	time_limit : int = None,
	num_generations : int = None,
	mutation_probability : tuple[int, int] = (0.5, 0.15)
):
	global global_time_limit, global_tests, global_model, animation_thread
	global_time_limit = time_limit
	global_tests = tests

	print("Started training model\n")

	ga_instance,_ = init(load, num_generations, mutation_probability)

	animation_thread = threading.Thread(target = runPlotAnimation)
	animation_thread.start()

	ga_instance.run()

	print("Finished training model\n")

	animation_thread.join()

	solution, solution_fitness, solution_idx = ga_instance.best_solution()
	for test in tests:
		if len(test) == 2:
			num_cards = test[0]
			bet_to_ante_ratio = test[1]
		elif len(test) == 3:
			num_cards = test[0]
			bet_to_ante_ratio = test[2] / test[1]
		global_model.set_weights(weights = pygad.kerasga.model_weights_as_matrix(model = global_model, weights_vector = solution))
		poker = PokerGame.newPokerStrategy(global_model, num_cards, bet_to_ante_ratio)
		PokerGame.printStrategy(poker)
		print()


	if save != None:
		ga_instance.save(save)
		print("Saved model\n")


def test():
	pass


# Note: Changing anything here will likely invalidate any models that have been saved
def init(load : str = None, num_generations : int = None, mutation_probability : tuple[int, int] = (0.5, 0.15)) -> tuple[pygad.GA, keras.Sequential]:
	global global_model
	
	NUM_SOLUTIONS = 50
	NUM_PARENTS_MATING = 15

	print()

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

	if load == None:
		if num_generations == None:
			num_generations = 10
		ga_instance = pygad.GA(
			num_generations = num_generations,
			fitness_func = fitness_func,
			num_parents_mating = NUM_PARENTS_MATING,
			initial_population = initial_population,
			save_solutions = True,
			parent_selection_type = "sss",
			mutation_type = "adaptive",
			crossover_type = "two_points",
			mutation_probability = mutation_probability,
		)
		print("Created GANN model\n")
	else:
		ga_instance = load(load)
		if num_generations != None:
			ga_instance.num_generations = num_generations
		print("Loaded GANN model\n")

	ga_instance.on_generation = on_generation
	ga_instance.mutation_probability = mutation_probability

	global_model = model

	return ga_instance, model


def runPlotAnimation():
	global ani, ax
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	ax.plot(plot_x_data, plot_y_data, 'b-')
	ax.set_title("Fitness over time")
	ax.set_xlabel("Generation")
	ax.set_ylabel("Fitness")
	ani = FuncAnimation(fig, plotAnimate, interval = 1000)
	pyplot.tight_layout()
	pyplot.show()


def plotAnimate(i):
	global ax
	plot_data_lock.acquire()
	ax.cla()
	ax.plot(plot_x_data, plot_y_data, 'b-')
	ax.set_title("Fitness over time")
	ax.set_xlabel("Generation")
	ax.set_ylabel("Fitness")
	plot_data_lock.release()


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
		strat = PokerGame.newPokerStrategy(model, num_cards, bet_to_ante_ratio)
		profits.append(PokerGame.computeExpectedWinnings(strat))
	return numpy.average(profits)


def fitness_func(solution, sol_idx):
	global global_tests, global_model
	return evaluate_solution(solution, global_model, global_tests)


last_gen_fitness = 0.0
num_successive_sols = 0
MAX_NUM_SUCCESSIVE_SOLS = 10
LOW_MUT = (0.5, 0.15)
HIGH_MUT = (0.85, 0.3)
LOW_CROSSOVER = 1.0
HIGH_CROSSOVER = 0.75
LOW_NUM_PARENTS = 15
HIGH_NUM_PARENTS = 15
def on_generation(ga_instance):
	global plot_x_data, plot_y_data, plot_data_lock, global_time_limit, last_gen_fitness, num_successive_sols
	print("Generation = {generation}".format(generation=ga_instance.generations_completed))
	fitness = ga_instance.best_solution()[1]
	print("Fitness    = {fitness}".format(fitness = fitness))
	print("Score      = {score}".format(score = min(1, max(0, fitness * 200 / 99) ) ** 5 ) )
	print()
	if math.isclose(fitness, last_gen_fitness, rel_tol=1e-10, abs_tol=0):
		num_successive_sols += 1
	else:
		num_successive_sols = 0
	last_gen_fitness = fitness
	global MAX_NUM_SUCCESSIVE_SOLS, LOW_MUT, HIGH_MUT, LOW_CROSSOVER, HIGH_CROSSOVER, LOW_NUM_PARENTS, HIGH_NUM_PARENTS
	if num_successive_sols > MAX_NUM_SUCCESSIVE_SOLS:
		ga_instance.mutation_probability = HIGH_MUT
		ga_instance.crossover_probability = HIGH_CROSSOVER
		ga_instance.num_parents_mating = HIGH_NUM_PARENTS
	else:
		ga_instance.mutation_probability = LOW_MUT
		ga_instance.crossover_probability = LOW_CROSSOVER
		ga_instance.num_parents_mating = LOW_NUM_PARENTS

	plot_data_lock.acquire()
	plot_x_data = numpy.linspace(0, len(ga_instance.best_solutions_fitness), num = len(ga_instance.best_solutions_fitness) + 1)
	plot_y_data = [0]
	for fitness in ga_instance.best_solutions_fitness:
		plot_y_data.append(fitness)
	plot_y_data = numpy.array(plot_y_data)
	plot_data_lock.release()
	if global_time_limit != None and seconds_elapsed() > global_time_limit:
		return "stop"
	if not animation_thread.is_alive():
		return "stop"


def seconds_elapsed():
	p = psutil.Process(os.getpid())
	return time.time() - p.create_time()


if __name__ == "__main__":
	main()