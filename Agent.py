
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
		self.update_frequency = args.update_frequency
		self.exps_prefix = args.exps_prefix
		self.rng = rng
		self.ddqn = ddqn
		self.memory_pool = memory_pool
		self.batch_size = args.batch_size
		self.replay_start_size = args.replay_start_size
		self.exps_prefix = args.exps_prefix
		self.valid_actions = valid_actions
		self.num_valid_actions = len(valid_actions)
		self.epsilon = self.epsilon_start
		self.episode_cnt = 0
		self.batch_cnt = 0
		self.epsilon_decay_rate = 0
		if self.epsilon_decay != 0:
			self.epsilon_decay_rate = (self.epsilon_start-self.epsilon_min)/self.epsilon_decay 

	def step(self,reward,phi,trainable):
		self.step_cnt += 1
		if trainable:
			if(self.memory_pool.curSz > self.replay_start_size):
				self.epsilon = max(self.epsilon_min,self.epsilon-self.epsilon_decay_rate)
			action = self.greedy_action(phi,self.epsilon,np.clip(reward,-1,1))
			if self.step_cnt % self.update_frequency == 0:
				self.batch_cnt += 1
				self.loss.append(self.train_my_model())
		else:
			self.episode_reward += reward
			action = self.greedy_action(phi,0.05,reward)
		self.__show_phi(self.last_phi,phi)
		self.last_phi = phi
		self.last_action = action
		return action
	
	def __show_phi(self,phi,phis):
		import matplotlib.pyplot as plt
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()


	def train_my_model(self):
		phi,action,reward,phis,terminal = self.memory_pool.stochasticSample(self.batch_size)
		self.ddqn.train(phi,action,reward,phis,terminal)

	def greedy_action(self,phi,epsilon,reward):
		self.memory_pool.add_sample(self.last_phi,self.last_action,reward,phi,False)
		if self.rng.rand() < epsilon:
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
		self.ddqn.TargetNetwork.save_weights(self.exps_prefix+"model_weight"+str(epoch)+".hdf5")
		
	def finish_episode(self,reward,terminal):
		pass

