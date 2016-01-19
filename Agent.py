
'''
game agent
'''
import logging
import numpy as np
import time
import uuid
class Agent:
	'''

	'''
	def __init__(self,args,ddqn,memory_pool,valid_actions,rng):

		self.epsilon_start = args.epsilon_start
		self.epsilon_min = args.epsilon_min
		self.epsilon_decay = args.epsilon_decay
		self.update_frequency = args.update_frequency
		self.exps_prefix = args.exps_prefix
		self.phi_len = args.phi_len
		self.holdout_data_size = args.holdout_data_size
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
		file_id = str(uuid.uuid4())
		self.training_file_path = self.exps_prefix+"training_results_"+file_id+".csv"
		self.testing_file_path = self.exps_prefix+"testing_results_"+self.file_id+".csv"
		self.open_training_result_file()
		self.open_testing_result_file()
		

	def open_testing_result_file(self):
		logging.info("Open"+self.testing_file_path)
		self.testing_result_file = open(testing_file_path,'w')
		self.testing_result_file.write('epoch,num_episodes,total_reward,reward_per_episode,mean_qval\n')
		self.testing_result_file.flush()

	def open_training_result_file(self):
		logging.info("Open"+self.training_file_path)
		self.training_result_file = open(self.training_file_path,'w')
		self.training_result_file.write('mean_loss,epsilon\n')
		self.training_result_file.flush()

	def update_training_result_file(self,mean_loss,epsilon):
		res = "{},{}\n".format(mean_loss,epsilon)
		self.training_result_file.write(res)
		self.training_result_file.flush()

	def update_testing_result_file(self,epoch,num_episodes,total_reward,reward_per_episode,mean_qval):
		res = "{},{},{},{},{}".format(epoch,num_episodes,total_reward,reward_per_episode,mean_qval)
		self.testing_result_file.write(res)
		self.testing_result_file.flush()

	def step(self,reward,phi,trainable):
		self.step_cnt += 1
		if trainable:
			if self.memory_pool.curSz > self.replay_start_size:
				self.epsilon = max(self.epsilon_min,self.epsilon-self.epsilon_decay_rate)
				if self.step_cnt % self.update_frequency == 0:
					self.batch_cnt += 1
					self.loss.append(self.train_my_model())
			action = self.greedy_action(phi,self.epsilon,np.clip(reward,-1,1))
			
		else:
			self.episode_reward += reward
			action = self.greedy_action(phi,0.05,reward)
	
		#self._show_phi(self.last_phi,phi)
		self.last_phi = phi
		self.last_action = action
		return action

	def _show_phi(self,phi,phis):
		'''
		This method is to test phi and phis
		'''
		import matplotlib.pyplot as plt
		for _ in xrange(self.phi_len):
			plt.subplot(2,self.phi_len,_+1)
			plt.imshow(phi[_],interpolation='none',cmap='gray')
		for _ in xrange(self.phi_len):
			plt.subplot(2,self.phi_len,_+5)
			plt.imshow(phis[_],interpolation='none',cmap='gray')
		plt.show()


	def train_my_model(self):
		phi,action,reward,phis,terminal = self.memory_pool.stochasticSample(self.batch_size)
		self.ddqn.train(phi,action,reward,phis,terminal)

	def greedy_action(self,phi,epsilon,reward):
		assert phi.shape[0] == self.phi_len
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
		self.trainable = True
		self.start_time = time.time()
		action = self.valid_actions[self.rng.randint(self.num_valid_actions)]
		self.last_action = action
		self.last_phi = phi
		return action
			
	def finish_epoch(self,epoch):
		self.ddqn.TargetNetwork.save_weights(self.exps_prefix+"model_weight"+str(epoch)+".hdf5")
		
	def finish_episode(self,reward,terminal,trainable):
		self.episode_reward += reward
		self.step_cnt += 1
		cost_time = time.time()-self.start_time
		if trainable:
			if self.batch_cnt > 0:
				mean_loss = np.mean(self.loss)
				self.update_training_result_file(mean_loss,self.epsilon)
				logging.info("average loss: {:.4f} and epsilon: {:.4f}".format(mean_loss,self.epsilon))
			if terminal:
				self.memory_pool.add_sample(self.last_phi,self.last_action,np.clip(reward,-1,1),self.last_phi,True)
		else:
			self.episode_cnt += 1
			self.test_reward += self.episode_reward

	def start_testing(self):
		self.trainable = False
		self.test_reward = 0
		self.episode_cnt = 0

	def finish_test(self,epoch):
		self.trainable = True
		if self.holdout_data is None and self.memory_pool.curSz > self.holdout_data_size:
			self.holdout_data = self.memory_pool.stochasticSample(self.holdout_data_size)[0]
		qval_mean = qval_sum = 0
		if self.holdout_data is not None:
			for _ in xrange(self.holdout_data_size):
				qval_sum += np.max(self.ddqn.predict(self.holdout_data[i]))
			qval_mean = qval_sum / self.holdout_data_size
		self.update_training_result_file(epoch,self.episode_cnt,self.test_reward,float(self.test_reward)/self.episode_cnt,qval_mean)

