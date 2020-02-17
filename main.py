#!/usr/bin/env python3

import sys
import re
import random as rd
import numpy as np
from itertools import combinations
from tqdm import tqdm


"""
./script.py rooms_file names_file constraints_file
rooms_file: <room_size>,<num_of_rooms>
names_file: <name>,<surname>,<gender (compared by ==)>
constraints_file: <index1>,<index2 or *>,<value>
"""

FIT_NO_CONSTRAINTS = 3
FIT_GENDER_MISSMATCH = 0
FIT_POWER = 1


""" Permutation array. """
class Genotype:
	def __init__(self, arr):
		self.arr = arr

	def __str__(self):
		return str(self.arr)

	def size(self):
		return len(self.arr)

	@staticmethod
	def random(num):
		g = Genotype(np.arange(num))
		np.random.shuffle(g.arr)
		return g

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
		if rd.randrange(0,1) < 0.1:
			return

		n = rd.randrange(0, round(perc * len(self.arr)))
		for _ in range(n):
			i1 = rd.randrange(len(self.arr))
			i2 = rd.randrange(len(self.arr))
			self.arr[i1], self.arr[i2] = self.arr[i2], self.arr[i1]


""" Pairwise cost matrix. """
class ValueMatrix:
	def __init__(self, constraints, names):
		self.genders = [n[2] for n in names]

		self.forall_matrix = {}
		self.matrix = {}
		for c in constraints:
			if c[1] == '*':  # Regex, imply same value for all pairs.
				self.forall_matrix[( int(c[0]) )] = float(c[2])
			else:
				self.matrix[( min(int(c[0]),int(c[1])), max(int(c[0]),int(c[1])) )] = float(c[2])  # Undirected graph.

	def get(self,i,j):
		# Explicit constraint applyed for all target values has most priority.
		x, y = self.forall_matrix.get(i), self.forall_matrix.get(j)
		if x or y:
			return min(x,y) if x and y else (x if x else y)  # Select lower value if both are present.

		v = self.matrix.get(( min(i,j), max(i,j) ))
		if v:  # Explicit constraints override implicit.
			return v
		else:  # Implicit values.
			if self.genders[i] != self.genders[j]:  # Male-female combinations are very discouraged.
				return FIT_GENDER_MISSMATCH

			return FIT_NO_CONSTRAINTS  # DEFAULT value for no connections.


class Evaluator:
	def load_array(self, filepath):
		arr = []
		try:
			with open(filepath, 'r') as f:
				for l in f:
					if not l.strip():  # Skip empty lines.
						continue
					parts = re.split('[\t,]', l.strip())
					arr.append(parts)
		except:
			print('Cannot parse file:',filepath)
			exit(1)
		return arr

	def __init__(self, rooms_file, names_file, constraints_file):
		self.rooms = self.load_array(rooms_file)  # <room_size>,<num_of_rooms>
		rooms = []
		for r in self.rooms:
			for _ in range(int(r[1])):
				rooms.append(int(r[0]))
		self.rooms = sorted(rooms)  # Incremental list of room sizes (repetitive).

		self.names = self.load_array(names_file)  # <name>,<surname>,<gender> (index is important)
		self.values = ValueMatrix(self.load_array(constraints_file), self.names)  # <index1>,<index2>,<value>

	def guest_count(self):
		return len(self.names)

	def evaluate(self, g):
		"""
			Inserts guests to rooms in genome order and accumulates the constraint values.
		"""
		value = 0

		i = 0
		for r in self.rooms:
			# Populate room.
			room = []
			for j in range(r):
				room.append(g.arr[i])
				i += 1
			
			# Calculate constraints value for all the pairs.
			v = 0
			ctr = 0
			for t in combinations(room,2):
				v += self.values.get(t[0],t[1])
				ctr += 1

			# Average value, so large rooms are same importance as small.
			value += v / ctr if ctr > 0 else v

		value = value ** FIT_POWER
		g.fitness = value
		return value

	def decode(self, g):
		"""
			Returns a string with a name per line and a room index in which guest sleeps.
		"""
		s = ''

		i = 0
		room_i = 0
		for r in self.rooms:
			for j in range(r):
				guest_i = g.arr[i]
				i += 1
				n = self.names[guest_i]
				s += '{},{},{}\n'.format(n[0], n[1], room_i)
			room_i += 1

		return s


def run(e, pop_size=100, max_iter=100):
	# Initialize population.
	pop = [Genotype.random(e.guest_count()) for _ in range(pop_size)]
	best = None  # Actual best.
	fit_sum = 0  # Fitness sum for probabilities.
	for g in pop:
		fit_sum += e.evaluate(g)
		if best is None or g.fitness > best.fitness:
				best = g

	print('Starting with best:', best.fitness)

	# The Loop.
	for i in tqdm(range(1,max_iter+1)):
		# Populate new population (elitism).
		new_pop = [best]
		new_sum = best.fitness
		for _ in range(pop_size-len(new_pop)):
			# Make a baby.
			parents = np.random.choice(pop, 2, p=[g.fitness/fit_sum for g in pop])
			child = Genotype.cross(parents[0], parents[1])
			child.mutate()
			# Add to population.
			new_sum += e.evaluate(child)
			new_pop.append(child)
			# Update global best.
			if child.fitness > best.fitness:
				best = child
				print('Iteration {} has best: {}'.format(i, best.fitness))
		# Set the new population.
		pop = new_pop
		fit_sum = new_sum

	return best


if __name__ == '__main__':
	evaluator = Evaluator(sys.argv[1], sys.argv[2], sys.argv[3])
	best = run(evaluator, pop_size=1000, max_iter=1000)
	
	result = evaluator.decode(best)

	print()
	print(result)
	with open('result.csv', 'w') as f:
		f.write(result)

