from asyncio import subprocess
from itertools import count
import string
import numpy
import numba
from numba import njit, jit
from tensorflow import keras
import os
import subprocess
from numba import int32, float32    # import the types
from numba.experimental import jitclass


@njit
def generatePredictionInputData(num_cards : int, bet_to_ante_ratio : int):
	predictions = numpy.empty((num_cards * 4, 3))
	curr = 0
	for i in range(1, num_cards + 1):
		predictions[curr] = numpy.array([i / num_cards, bet_to_ante_ratio, 0.5])
		curr += 1
		predictions[curr] = numpy.array([i / num_cards, bet_to_ante_ratio, 0.25])
		curr += 1
		predictions[curr] = numpy.array([i / num_cards, bet_to_ante_ratio, 1.0])
		curr += 1
		predictions[curr] = numpy.array([i / num_cards, bet_to_ante_ratio, 0.75])
		curr += 1
	return predictions


def newPokerStrategy(model : keras.Sequential, num_cards : int, bet_to_ante_ratio : int):
	# predictions = numpy.empty((num_cards * 4, 3))
	predictions = model.predict(generatePredictionInputData(num_cards, bet_to_ante_ratio))
	return PokerStrategy(num_cards, bet_to_ante_ratio, predictions)


dict_spec = (numba.types.int32, numba.types.boolean)
spec = [
	('chance_to_open_check', float32[:]),
	('chance_to_check_check', float32[:]),
	('chance_to_bet_fold', float32[:]),
	('chance_to_check_bet_fold', float32[:]),
	('num_cards', int32),
	('bet_to_ante_ratio', float32),
	('author_checks_to_open_check', numba.types.DictType(*dict_spec)),
	('author_folds_to_open_bet', numba.types.DictType(*dict_spec)),
	('author_folds_to_check_bet', numba.types.DictType(*dict_spec)),
	('author_open_checks', numba.types.DictType(*dict_spec)),
]

@jitclass(spec)
class PokerStrategy:
	def __init__(self, num_cards : int, bet_to_ante_ratio : int, predictions : numba.float32[:, :]): # numpy.ndarray
		self.num_cards = num_cards
		self.bet_to_ante_ratio = bet_to_ante_ratio
		self.chance_to_open_check = numpy.zeros(num_cards + 1, numba.float32)
		self.chance_to_check_check = numpy.zeros(num_cards + 1, numba.float32)
		self.chance_to_bet_fold = numpy.zeros(num_cards + 1, numba.float32)
		self.chance_to_check_bet_fold = numpy.zeros(num_cards + 1, numba.float32)

		self.author_checks_to_open_check = numba.typed.Dict.empty(*dict_spec)
		self.author_folds_to_open_bet = numba.typed.Dict.empty(*dict_spec)
		self.author_folds_to_check_bet = numba.typed.Dict.empty(*dict_spec)
		self.author_open_checks = numba.typed.Dict.empty(*dict_spec)

		curr = 0
		for i in range(1, self.num_cards + 1):
			self.chance_to_open_check[i] = (predictions[curr][0])
			curr += 1
			self.chance_to_check_check[i] = (predictions[curr][0])
			curr += 1
			self.chance_to_bet_fold[i] = (predictions[curr][0])
			curr += 1
			self.chance_to_check_bet_fold[i] = (predictions[curr][0])
			curr += 1
	# def __init__(self, model : keras.Sequential, num_cards : int, bet_to_ante_ratio : int):
	# 	self.chance_to_open_check = [0]
	# 	self.chance_to_check_check = [0]
	# 	self.chance_to_bet_fold = [0]
	# 	self.chance_to_check_bet_fold = [0]
	# 	self.num_cards = num_cards
	# 	self.bet_to_ante_ratio = bet_to_ante_ratio

	# 	for i in range(1, self.num_cards + 1):
	# 		self.chance_to_open_check.append( \
	# 			model.predict(numpy.array([[i / num_cards, bet_to_ante_ratio, 0.5]]))[0][0]
	# 		)
	# 		self.chance_to_check_check.append( \
	# 			model.predict(numpy.array([[i / num_cards, bet_to_ante_ratio, 0.25]]))[0][0]
	# 		)
	# 		self.chance_to_bet_fold.append( \
	# 			model.predict(numpy.array([[i / num_cards, bet_to_ante_ratio, 1]]))[0][0]
	# 		)
	# 		self.chance_to_check_bet_fold.append( \
	# 			model.predict(numpy.array([[i / num_cards, bet_to_ante_ratio, 0.75]]))[0][0]
	# 		)


	def __str__(self) -> string:
		ans = ""
		for card in range(1, self.num_cards + 1):
			ans += str(card) + " open: check " + str(self.chance_to_open_check[card]) + "\n"
			ans += str(card) + " check: check " + str(self.chance_to_check_check[card]) + "\n"
			ans += str(card) + " bet: fold " + str(self.chance_to_bet_fold[card]) + "\n"
			ans += str(card) + " check-bet: fold " + str(self.chance_to_check_bet_fold[card]) + "\n"
		return ans

	# def output_to_files(self, file : string = "poker"):
	# 	text_file = open(os.getcwd() + "/" + file + ".in", 'w');
	# 	text_file.write(str(self.num_cards) + " " + str(1) + " " + str(self.bet_to_ante_ratio))
	# 	text_file.close()
	# 	text_file = open(os.getcwd() + "/" + file + ".out", 'w');
	# 	text_file.write(str(self))
	# 	text_file.close()
	

	# def compute_expected_profit(self) -> float:
	# 	self.output_to_files()
	# 	return float(subprocess.check_output([os.getcwd() + "/checker.exe", "./poker.in", "--out", "./poker.out"]).decode())
	

	def __antePotWinnings(self, self_card : int, opponents_card : int) -> float:
		if self_card > opponents_card:
			return 1.0
		return 0.0
	

	def __betPotWinnings(self, self_card : int, opponents_card : int) -> float:
		if self_card > opponents_card:
			return 1 + self.bet_to_ante_ratio
		return -self.bet_to_ante_ratio


	def computeExpectedWinnings(self) -> float:
		total_winnings = 0.0
		total_situations = 0
		for my_card in range(1, self.num_cards + 1):
			for authors_card in range(1, self.num_cards + 1):
				if my_card == authors_card:
					continue
				for I_am_first in [True, False]:
					if I_am_first: # open
						# check
						chance_to_open_check = self.chance_to_open_check[my_card]
						winnings_if_open_check = 0.0
						if self.__authorChecksToOpenCheck(authors_card): # author checks
							winnings_if_open_check += self.__antePotWinnings(my_card, authors_card)
						else: # author bets (if I fold I don't lose anything)
							winnings_if_open_check \
								+= (1 - self.chance_to_check_bet_fold[my_card]) \
								* self.__betPotWinnings(my_card, authors_card)
						# bet
						chance_to_open_bet = 1 - chance_to_open_check
						winnings_if_open_bet = 0.0
						if self.__authorFoldsToOpenBet(authors_card): # author folds
							winnings_if_open_bet += 1.0
						else: # author calls
							winnings_if_open_bet += self.__betPotWinnings(my_card, authors_card)
						# total
						total_winnings \
							+= chance_to_open_check * winnings_if_open_check \
							+ chance_to_open_bet * winnings_if_open_bet
					else: # author is first
						if self.__authorOpenChecks(authors_card): # author checks
							# check check
							chance_to_check_check = self.chance_to_check_check[my_card]
							winnings_if_check_check = 0.0
							winnings_if_check_check += self.__antePotWinnings(my_card, authors_card)
							# check bet
							chance_to_check_bet = 1 - chance_to_check_check
							winnings_if_check_bet = 0.0
							if self.__authorFoldsToCheckBet(authors_card):
								winnings_if_check_bet += 1
							else:
								winnings_if_check_bet += self.__betPotWinnings(my_card, authors_card)
							# total
							total_winnings \
								+= chance_to_check_check * winnings_if_check_check \
								+ chance_to_check_bet * winnings_if_check_bet
						else: # author bets
							# bet fold
							chance_to_bet_fold = self.chance_to_bet_fold[my_card]
							winnings_if_bet_fold = 0.0 # we don't lose anything here
							# bet call
							chance_to_bet_call = 1 - chance_to_bet_fold
							winnings_if_bet_call = 0.0
							winnings_if_bet_call = self.__betPotWinnings(my_card, authors_card)
							# total
							total_winnings \
								+= chance_to_bet_fold * winnings_if_bet_fold \
								+ chance_to_bet_call * winnings_if_bet_call
					total_situations += 1
		return total_winnings / total_situations
	

	def __authorOpenChecks(self, authors_card : int) -> bool:
		if authors_card in self.author_open_checks:
			return self.author_open_checks[authors_card]
		
		expected_winnings_if_check = 0.0
		expected_winnings_if_bet = 0.0

		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			# author bets
			expected_winnings_if_bet \
				+= 1 * (self.chance_to_bet_fold[potential_my_card]) \
				+ \
					self.__betPotWinnings(authors_card, potential_my_card) \
					* (1 - self.chance_to_bet_fold[potential_my_card])
			# author checks
			if self.__authorFoldsToCheckBet(authors_card):
				expected_winnings_if_check \
					+= \
						self.__antePotWinnings(authors_card, potential_my_card) \
						* self.chance_to_check_check[potential_my_card] \
					+ \
						self.__betPotWinnings(0, potential_my_card) \
						* (1 - self.chance_to_check_check[potential_my_card])
			else:
				expected_winnings_if_bet \
					+= \
						1 * self.chance_to_check_bet_fold[authors_card] \
					+ \
						self.__betPotWinnings(authors_card, potential_my_card) \
						* (1 - self.chance_to_check_bet_fold[authors_card])

		self.author_open_checks[authors_card] = expected_winnings_if_check > expected_winnings_if_bet
		return self.author_open_checks[authors_card]



	def __authorChecksToOpenCheck(self, authors_card : int) -> bool:
		if authors_card in self.author_checks_to_open_check:
			return self.author_checks_to_open_check[authors_card]
		
		chance_to_have_open_checked = 0.0
		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			chance_to_have_open_checked += self.chance_to_open_check[potential_my_card]
		
		expected_winnings_if_check = 0.0
		expected_winnings_if_bet = 0.0
		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			expected_winnings_if_check \
				+= self.__antePotWinnings(authors_card, potential_my_card) \
				* (self.chance_to_open_check[potential_my_card] / chance_to_have_open_checked)
			expected_winnings_if_bet \
				+= (
					self.chance_to_check_bet_fold[potential_my_card] * 1 # I fold
					+ (1 - self.chance_to_check_bet_fold[potential_my_card]) # I call
					* self.__betPotWinnings(authors_card, potential_my_card)
				) \
				* (self.chance_to_open_check[potential_my_card] / chance_to_have_open_checked)
		
		self.author_checks_to_open_check[authors_card] = expected_winnings_if_check > expected_winnings_if_bet
		return self.author_checks_to_open_check[authors_card]
	

	def __authorFoldsToOpenBet(self, authors_card : int) -> bool:
		if authors_card in self.author_folds_to_open_bet:
			return self.author_folds_to_open_bet[authors_card]
		
		chance_to_have_open_bet = 0.0
		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			chance_to_have_open_bet += (1 - self.chance_to_open_check[potential_my_card])
		
		expected_winnings_if_fold = 0.0
		expected_winnings_if_call = 0.0

		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			expected_winnings_if_call \
				+= self.__betPotWinnings(authors_card, potential_my_card) \
				* ((1 - self.chance_to_open_check[potential_my_card]) / chance_to_have_open_bet)

		self.author_folds_to_open_bet[authors_card] = expected_winnings_if_fold > expected_winnings_if_call
		return self.author_folds_to_open_bet[authors_card]
	

	def __authorFoldsToCheckBet(self, authors_card : int) -> bool:
		if authors_card in self.author_folds_to_check_bet:
			return self.author_folds_to_check_bet[authors_card]

		chance_to_have_check_bet = 0.0
		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			chance_to_have_check_bet += (1 - self.chance_to_check_check[potential_my_card])
		
		expected_winnings_if_fold = 0.0
		expected_winnings_if_call = 0.0

		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			expected_winnings_if_call \
				+= self.__betPotWinnings(authors_card, potential_my_card) \
				* ((1 - self.chance_to_check_check[potential_my_card]) / chance_to_have_check_bet)

		self.author_folds_to_check_bet[authors_card] = expected_winnings_if_fold > expected_winnings_if_call
		return self.author_folds_to_check_bet[authors_card]

def strink(_self):
	ans = ""
	for card in range(1, _self.num_cards + 1):
		ans += str(card) + " open: check " + str(_self.chance_to_open_check[card]) + "\n"
		ans += str(card) + " check: check " + str(_self.chance_to_check_check[card]) + "\n"
		ans += str(card) + " bet: fold " + str(_self.chance_to_bet_fold[card]) + "\n"
		ans += str(card) + " check-bet: fold " + str(_self.chance_to_check_bet_fold[card]) + "\n"
	return ans

def output_to_files(_self, file : string = "poker"):
	text_file = open(os.getcwd() + "/" + file + ".in", 'w');
	text_file.write(str(_self.num_cards) + " " + str(1) + " " + str(_self.bet_to_ante_ratio))
	text_file.close()
	text_file = open(os.getcwd() + "/" + file + ".out", 'w');
	text_file.write(strink(_self))
	text_file.close()


def compute_expected_profit(_self) -> float:
	output_to_files(_self)
	return float(subprocess.check_output([os.getcwd() + "\\checker.exe", "./poker.in", "--out", "./poker.out"]).decode())