import random

def random_step(obs):
	return random.randint(0,1)

def naive_angle(obs):
	angle = obs[2]
	return 0 if angle < 0 else 1
