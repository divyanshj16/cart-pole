import random

def naive_constant(obs):
	return 0

def random_step(obs):
	return random.randint(0,1)

def naive_angle(obs):
	angle = obs[2]
	return 0 if angle < 0 else 1

def naive_pos(obs): # trying to keep rod at the center
	pos = obs[0] # positiion
	return 0 if pos > 0 else 1

def naive_vel(obs):
	vel = obs[1] # velocity
	return 0 if vel > 0 else 1

def naive_vel_pos(obs):
	pos = obs[0] # positiion
	vel = obs[1] # velocity
	temp = pos * vel # anything random
	return 0 if temp > 0 else 1




