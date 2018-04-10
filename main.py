import gym
from naive import *
import numpy as np
import argparse

env = gym.make('CartPole-v0')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Control internal variables RENDER, EPISODES etc.')
	parser.add_argument('--render', action='store_true', help='pass this argument to render', default=False)
	args = parser.parse_args()

	EPISODES = 500
	MAX_STEPS = 1000
	FN = naive_constant
	RENDER = args.render
	PRINT_EVERY = 100

	if PRINT_EVERY is None:
		PRINT_EVERY = EPISODES + 1

	if RENDER:
		EPISODES = 1

	a_reward = [] # total summed rewards of every episode
	a_t = []      # step before ending of an episode

	for episode in range(EPISODES):
		if episode%PRINT_EVERY == 0:
			print(f'{episode} episodes completed')
		e_reward = 0
		observation = env.reset()
		rnd_ctr = 0
		for t in range(MAX_STEPS):	
			if RENDER:
				env.render()
			step = FN(observation)
			observation , reward, done, info = env.step(step)
			e_reward += reward
			if done:
				if not RENDER:
					if episode%PRINT_EVERY == 0:
						print(f'Ended at timestep {t}')
					a_t.append(t)
					break
				else:
					rnd_ctr += 1
					print(done,rnd_ctr)
					if rnd_ctr == 50:
						break
		a_reward.append(e_reward)

	if not RENDER:
		print(f' The mean score over {EPISODES} episodes for average {np.mean(a_t)} steps is {np.mean(a_reward)}.')
		print(f' STD: {np.std(a_t)} steps , {np.std(a_reward)} rewards')
		print(f' MAX: {np.max(a_t)} steps , MIN:  {np.min(a_t)} steps')




