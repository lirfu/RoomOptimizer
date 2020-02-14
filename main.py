#!/usr/bin/env python3

import sys
import re
import numpy as np


# Get the params.
rooms_file = sys.argv[1]
names_file = sys.argv[2]
constraints_file = sys.argv[3]


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

rooms = load_array(rooms_file)  # <num_of_rooms>,<room_size>
names = load_array(names_file)  # <name>,<surname> (index shows )
constraints = load_array(constraints_file)


# 

