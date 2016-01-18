
'''
game agent
'''
import logging
import numpy as np
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

		if(self.memory_pool.curSz > self.replay_start_size):
			pass
		else:

			pass


	def __train_my_model(self):
	'''
	train my network and return loss
	'''	
		phi,action,reward,phis,terminal = self.memory_pool(self.batch_size)
		return self.train(phi,action,reward,phis,terminal)
		

	def greedy_action(self,phi):
	'''
	choose the epsilon-greedy action
	'''
		if rng.random() < self.epsilon:
			action = self.ddqn.choose_action(phi)
		else:
			action = rng.randint(0,self.num_valid_actions)
		
		return action
	def start_episode(self,phi):
		self.step_cnt = 0
		self.loss = []
		pass
			
	def finish_epoch(self,epoch):
		pass
		#TODO: save model weights


