import pygad
import PokerGame
import numpy
import random


class PokerGA:
	def __init__(
		self,
		num_cards : int,
		bet_to_ante_ratio : int,
		num_generations : int = 20,
		sol_per_pop : int = 50,
		num_parents_mating : int = 10,
		parent_selection_type : str = "sss",
		crossover_type : str = "two_points",
		mutation_type : str = "adaptive",
		mutation_probability = (0.4, 0.1),
		crossover_probability = 1.0,
		max_stagnant_solutions = -1,

		load : str = None,
		enforce_load_name_standard : bool = False,
	) -> None:
		self.num_cards = num_cards
		self.bet_to_ante_ratio = bet_to_ante_ratio

		if max_stagnant_solutions == -1:
			init_mutation_probability = mutation_probability
			init_crossover_probability = crossover_probability
		else:
			init_mutation_probability = mutation_probability[0]
			init_crossover_probability = crossover_probability[0]

		self.ga_instance = pygad.GA(
			num_generations = num_generations,
			num_genes = 4 * num_cards,
			gene_space = { "low" : 0.0, "high" : 1.0 },
			sol_per_pop = sol_per_pop,
			num_parents_mating = num_parents_mating,
			parent_selection_type = parent_selection_type,
			crossover_type = crossover_type,
			mutation_type = mutation_type,
			fitness_func = fitness_func,
			on_generation = on_generation,
			mutation_probability = init_mutation_probability,
			crossover_probability = init_crossover_probability,
			random_mutation_min_val = -0.05,
			random_mutation_max_val = 0.05,
		)


def fitness_func(solution, sol_idx):
	global poker_ga
	strat = PokerGame.PokerStrategy(poker_ga.num_cards, poker_ga.bet_to_ante_ratio, numpy.array(solution))
	return PokerGame.computeExpectedWinnings(strat)


def on_generation(ga_instance):
	if ga_instance.generations_completed % 100 == 0:
		print("Generation = {generation}".format(generation=ga_instance.generations_completed))
		fitness = ga_instance.best_solution()[1]
		print("Fitness    = {fitness}".format(fitness = fitness))
		print("Score      = {score}".format(score = min(1, max(0, fitness * 200 / 99) ) ** 5 ) )


poker_ga : PokerGA = PokerGA(
	31, 5 / 4,
	num_generations = 10000,
	num_parents_mating = 5,
	crossover_type = "uniform",
	mutation_type = "random",
	mutation_probability = 0.5,
	sol_per_pop = 10,
	crossover_probability = 0.5
)
# poker_ga.ga_instance.run()


init_pop = numpy.array([random.uniform(0.0, 1.0) for i in range(0, 4 * 31)])
poker_strat = PokerGame.PokerStrategy(31, 5/4, init_pop)
best = -1

for i in range(0, 100000):
	card = random.randint(1, 31)
	sit = random.randint(1, 4)
	match sit:
		case 1:
			old = poker_strat.chance_to_open_check[card]
		case 2:
			old = poker_strat.chance_to_check_check[card]
		case 3:
			old = poker_strat.chance_to_bet_fold[card]
		case 4:
			old = poker_strat.chance_to_check_bet_fold[card]
	
	delta = 1e3 / (i + 1)
	new = old + random.uniform(-delta, delta)

	match sit:
		case 1:
			poker_strat.chance_to_open_check[card] = new
		case 2:
			poker_strat.chance_to_check_check[card] = new
		case 3:
			poker_strat.chance_to_bet_fold[card] = new
		case 4:
			poker_strat.chance_to_check_bet_fold[card] = new

	if poker_strat.chance_to_open_check[card] < 0 \
	or poker_strat.chance_to_open_check[card] > 1 \
	or poker_strat.chance_to_check_check[card] < 0 \
	or poker_strat.chance_to_check_check[card] > 1 \
	or poker_strat.chance_to_bet_fold[card] < 0 \
	or poker_strat.chance_to_bet_fold[card] > 1 \
	or poker_strat.chance_to_check_bet_fold[card] < 0 \
	or poker_strat.chance_to_check_bet_fold[card] > 1:
		match sit:
			case 1:
				poker_strat.chance_to_open_check[card] = old
			case 2:
				poker_strat.chance_to_check_check[card] = old
			case 3:
				poker_strat.chance_to_bet_fold[card] = old
			case 4:
				poker_strat.chance_to_check_bet_fold[card] = old
		continue

	new_thing = PokerGame.compute_expected_profit(poker_strat) # PokerGame.computeExpectedWinnings(poker_strat)
	# if(new_thing > best):
	# 	best = new_thing
	print(new_thing)
	print(PokerGame.computeExpectedWinnings(poker_strat))
	print()
	# else:
	match sit:
		case 1:
			poker_strat.chance_to_open_check[card] = old
		case 2:
			poker_strat.chance_to_check_check[card] = old
		case 3:
			poker_strat.chance_to_bet_fold[card] = old
		case 4:
			poker_strat.chance_to_check_bet_fold[card] = old



print(PokerGame.toString(poker_strat))