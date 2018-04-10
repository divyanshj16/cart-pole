import numpy as np

def discount_rewards(rewards,drate = 0.8):
	l = len(rewards)
	rewards = np.array(rewards)
	drates = np.array([drate**i for i in range(l)])
	dreward = np.array([np.sum(rewards[i:] * drates[:l-i]) for i in range(l)])
	return dreward

def normalize_rew(all_rewards, drate = 0.8):
	# pdb.set_trace()
	all_discounted_rewards = [discount_rewards(rewards) for rewards in all_rewards]
	flat_rewards = np.concatenate(all_discounted_rewards)
	reward_mean = flat_rewards.mean()
	reward_std = flat_rewards.std()
	return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]