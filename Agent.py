
'''
game agent
'''
import logging
import numpy as np
import time
class Agent:
	'''

	'''
	def __init__(self,args,ddqn,memory_pool,valid_actions,rng):

		self.epsilon_start = args.epsilon_start
		self.epsilon_min = args.epsilon_min
		self.epsilon_decay = args.epsilon_decay
		self.rng = rng
		self.ddqn = ddqn
		self.memory_pool = memory_pool
		self.batch_size = args.batch_size
		self.replay_start_size = args.replay_start_size

		self.valid_actions = valid_actions
		self.num_valid_actions = len(valid_actions)
		self.epsilon = self.epsilon_start
		self.episode_cnt = 0
		self.batch_cnt = 0
	def step(self,reward,phi):
		self.step_cnt += 1

		#TODO:update self.epsilon

		#if(self.memory_pool.curSz > self.replay_start_size):
			#None
	def train_my_model(self):
		phi,action,reward,phis,terminal = self.memory_pool.stochasticSample(self.batch_size)
		self.ddqn.train(phi,action,reward,phis,terminal)


	def greedy_action(self,phi):

		if self.rng.rand() < self.epsilon:
			action = self.valid_actions[self.rng.randint(0,self.num_valid_actions)]
		else:
			action = self.ddqn.choose_action(phi)
		return action
	def start_episode(self,phi):
		self.step_cnt = 0
		self.batch_cnt = 0
		self.episode_reward = 0
		self.loss = []
		self.start_time = time.time()
		action = self.valid_actions[self.rng.randint(self.num_valid_actions)]
		self.last_action = action
		self.last_phi = phi
		return action
			
	def finish_epoch(self,epoch):
		pass
		#TODO: save model weights
	def finish_episode(self,reward,terminal):
		pass

