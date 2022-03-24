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


spec = [
	('chance_to_open_check', float32[:]),
	('chance_to_check_check', float32[:]),
	('chance_to_bet_fold', float32[:]),
	('chance_to_check_bet_fold', float32[:]),
	('num_cards', int32),
	('bet_to_ante_ratio', float32),
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


	def compute_average_winnings(self) -> float:
		avg_winnings = 0.0

		for my_card in range(1, self.num_cards + 1):
			# open
			winnings_if_open = 0.0
			winnings_if_open_check = 0.0
			chance_to_open_check = self.chance_to_open_check[my_card]
			winnings_if_open_bet = 0.0
			chance_to_open_bet = 1 - chance_to_open_check
			for authors_card in range(1, self.num_cards + 1):
				if authors_card == my_card:
					continue
				# open check
				if self.author_checks_to_open_check(authors_card):
					if my_card > authors_card:
						winnings_if_open_check += 1
					else:
						winnings_if_open_check -= 1
				else:
					winnings_if_open_check_bet_fold = 0.0
					chance_to_open_check_bet_fold = self.chance_to_check_bet_fold[my_card]
					winnings_if_open_check_bet_call = 0.0
					chance_to_open_check_bet_call = 1 - chance_to_open_check_bet_fold
					winnings_if_open_check_bet_fold -= 1
					if my_card > authors_card:
						winnings_if_open_check_bet_call += 1 + self.bet_to_ante_ratio
					else:
						winnings_if_open_check_bet_call -= 1 + self.bet_to_ante_ratio
					winnings_if_open_check \
						+= winnings_if_open_check_bet_fold * chance_to_open_check_bet_fold \
						+ winnings_if_open_check_bet_call * chance_to_open_check_bet_call
				# open bet
				if self.author_folds_to_open_bet(authors_card):
					winnings_if_open_bet += 1
				else:
					if my_card > authors_card:
						winnings_if_open_bet += 1 + self.bet_to_ante_ratio
					else:
						winnings_if_open_bet -= 1 + self.bet_to_ante_ratio
			winnings_if_open_check /= self.num_cards - 1
			winnings_if_open_bet /= self.num_cards - 1
			winnings_if_open \
				+= winnings_if_open_check * chance_to_open_check \
				+ winnings_if_open_bet * chance_to_open_bet

			# check
			winnings_if_check = 0.0
			winnings_if_check_check = 0.0
			winnings_if_check_bet = 0.0
			chance_to_check_check = self.chance_to_check_check[my_card]
			chance_to_check_bet = 1 - chance_to_check_check
			for authors_card in range(1, self.num_cards + 1):
				if authors_card == my_card:
					continue
				# check check
				if my_card > authors_card:
					winnings_if_check_check += 1
				else:
					winnings_if_check_check -= 1
				# check bet
				if self.author_folds_to_check_bet:
					winnings_if_check_bet += 1
				else:
					pot = 1 + self.bet_to_ante_ratio
					if my_card > authors_card:
						winnings_if_check_bet += pot
					else:
						winnings_if_check_bet -= pot
			winnings_if_check_check /= self.num_cards - 1
			winnings_if_check_bet /= self.num_cards - 1
			winnings_if_check \
				+= winnings_if_check_check * chance_to_check_check \
				+ winnings_if_check_bet * chance_to_check_bet
			
			# bet
			winnings_if_bet = 0.0
			winnings_if_bet_fold = 0.0
			winnings_if_bet_call = 0.0
			chance_to_bet_fold = self.chance_to_bet_fold[my_card]
			chance_to_bet_call = 1 - chance_to_bet_fold
			for authors_card in range(1, self.num_cards + 1):
				if authors_card == my_card:
					continue
				# bet fold
				winnings_if_bet_fold -= 1
				# bet call
				if my_card > authors_card:
					winnings_if_bet_call += 1 + self.bet_to_ante_ratio
				else:
					winnings_if_bet_call -= 1 + self.bet_to_ante_ratio
			winnings_if_bet_fold /= self.num_cards - 1
			winnings_if_bet_call /= self.num_cards - 1
			winnings_if_bet \
				+= winnings_if_bet_fold * chance_to_bet_fold \
				+ winnings_if_bet_call * chance_to_bet_call

			avg_winnings += numpy.average([winnings_if_open, numpy.average([winnings_if_check, winnings_if_bet])])
		
		avg_winnings /= self.num_cards
		return avg_winnings
	

	def author_checks_to_open_check(self, authors_card : int) -> bool:
		chance_to_have_checked = 0.0
		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			chance_to_have_checked += self.chance_to_open_check[potential_my_card]
		winnings_if_check = 0.0
		winnings_if_bet = 0.0
		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			if authors_card > potential_my_card:
				winnings_if_check \
					+= 1 * (self.chance_to_open_check[potential_my_card] / chance_to_have_checked)
				winnings_if_bet \
					+= (
						1 * self.chance_to_bet_fold[potential_my_card]
						+ (1 + self.bet_to_ante_ratio) * (1 - self.chance_to_bet_fold[potential_my_card])
					) \
					* (self.chance_to_open_check[potential_my_card] / chance_to_have_checked)
			else:
				winnings_if_check \
					-= 1 * (self.chance_to_open_check[potential_my_card] / chance_to_have_checked)
				winnings_if_bet \
					-= (
						1 * self.chance_to_bet_fold[potential_my_card]
						+ (1 + self.bet_to_ante_ratio) * (1 - self.chance_to_bet_fold[potential_my_card])
					) \
					* (self.chance_to_open_check[potential_my_card] / chance_to_have_checked)
		return winnings_if_check > winnings_if_bet
	

	def author_folds_to_open_bet(self, authors_card : int) -> bool:
		chance_to_have_bet = 0.0
		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			chance_to_have_bet += (1 - self.chance_to_open_check[potential_my_card])
		
		winnings_if_fold = 0.0
		winnings_if_call = 0.0

		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			winnings_if_fold -= 1 * ((1 - self.chance_to_open_check[potential_my_card]) / chance_to_have_bet)
			if authors_card > potential_my_card:
				winnings_if_call += (1 + self.bet_to_ante_ratio) \
					* ((1 - self.chance_to_open_check[potential_my_card]) / chance_to_have_bet)
			else:
				winnings_if_call -= (1 + self.bet_to_ante_ratio) \
					* ((1 - self.chance_to_open_check[potential_my_card]) / chance_to_have_bet)

		return winnings_if_fold > winnings_if_call
	

	def author_folds_to_check_bet(self, authors_card : int) -> bool:
		chance_to_have_check_bet = 0.0
		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			chance_to_have_check_bet += (1 - self.chance_to_check_check[potential_my_card])
		
		winnings_if_fold = 0.0
		winnings_if_call = 0.0

		for potential_my_card in range(1, self.num_cards + 1):
			if potential_my_card == authors_card:
				continue
			winnings_if_fold -= 1 * ((1 - self.chance_to_check_check[potential_my_card]) / chance_to_have_check_bet)
			pot = (1 + self.bet_to_ante_ratio) \
				* ((1 - self.chance_to_check_check[potential_my_card]) / chance_to_have_check_bet)
			if authors_card > potential_my_card:
				winnings_if_call += pot
			else:
				winnings_if_call -= pot

		return winnings_if_fold > winnings_if_call

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