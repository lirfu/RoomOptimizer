#!/usr/bin/env python3

import sys
import re
import random as rd
import numpy as np


# Load data.
def load_array(filepath):
	arr = []
	try:
		with open(filepath, 'r') as f:
			for l in f:
				if not l.strip():  # Skip empty lines.
					continue
				parts = re.split(l.strip(), '[\t,]')
				arr.append(parts)
	except:
		print('Cannot parse file:',filepath)
		exit(1)
	return arr


# GA stuff
class Genotype:
	def __init__(self, arr):
		self.arr = arr

	@classmethod
	def random(self, num):
		return self(np.random.shuffle(np.arange(num)))

	def size(self):
		return len(self.arr)

	@staticmethod
	def cross(g1, g2):
		if rd.getrandbits(1) == 0:
			control = g1
			target = g2
		else:
			control = g2
			target = g1
		child = []
		for i in control.arr:
			child.append(target.arr[i])
		return Genotype(child)

	def mutate(self, perc=0.3):
		n = rd.randrange(0, perc * len(self.arr))
		for _ in range(n):
			i1 = rd.randrange(len(self.arr))
			i2 = rd.randrange(len(self.arr))
			self.arr[i1], self.arr[i2] = self.arr[i2], self.arr[i1]


class Evaluator:
	def __init__(self, rooms_file, names_file, constraints_file):
		self.rooms = load_array(rooms_file)  # <num_of_rooms>,<room_size>
		self.names = load_array(names_file)  # <name>,<surname> (index shows )
		self.constraints = load_array(constraints_file)

	def guest_count():
		return len(names)

	def evaluate(g):
		"""
			Inserts guests to rooms in genome order and accumulates the constraint values.
		"""
		pass  # TODO

	def decode(g):
		"""
			Returns a string with a name per line and a room index in which guest sleeps.
		"""
		pass  # TODO


def run(pop_size=100, max_iter=100):
	pop = [Genotype.random(pop_size) for _ in range(pop_size)]

	def cmp(g1, g2):
		return g1.fitness - g2.fitness

	for i in range(1,max_iter+1):
		new_pop = []
		best = None

		for _ in range(pop_size):
			parents = np.random.choice(pop, 2, p=[g.fitness for g in pop])
			child = Genotype.cross(parents[0], parents[1])
			child.mutate()
			evaluate(child)
			new_pop.append(child)
			if best is None or child.fitness > best.fitness:
				best = child

		print('Iteration {} has best: {}'.format(i, best.fitness))

	return best


if __name__ == '__main__':
	evaluator = Evaluator(sys.argv[1], sys.argv[2], sys.argv[3])
	best = run(evaluator)
	
	result = evaluator.decode(best)
	print(result)

	with open('result.csv', 'wb') as f:
		f.write(result)

