
import copy
import numpy as np
import logging
'''
double deep Q-learning network
based on keras,a python-based deep neural network framework
'''

class DDQN:
	def __init__(self,network,valid_actions,target_nn_update_frequency,discount):
		#Target network chooses the action
		#Online network eval the Q-val
		self.OnlineNetwork = network
		self.TargetNetwork = network
		self.valid_actions = valid_actions
		self.discount = discount
		self.target_nn_update_frequency = target_nn_update_frequency
		self.num_valid_actions = len(valid_actions)
		self.train_cnt = 0
	def choose_action(self,phi):
		action_idx = self.predictAction(phi)
		return self.valid_actions[action_idx]

	def train(self,phi1,phi2,act,reward,terminals):
		assert phi1.shape == phi2.shape
		n_samples = phi1.shape[0]
		#qval1 = self.predictQval(phi1)
		qval2 = self.predictQval(phi2)
		Y = []

		for _ in xrange(n_samples):		
			if terminals[_]:
				y = reward[_]
			else:
				action_idx = self.predictAction(phi2[_])
				y = reward[_] + self.discount*qval2[_][action_idx]
			Y.append(y)
		Y = np.asarray(Y)
		loss = self.OnlineNetwork.train_on_batch(phi1,Y)
		self.train_cnt += 1
		if self.train_cnt % self.target_nn_update_frequency == 0:
			self.update()
		return loss

	def predictAction(self,phi):
		assert len(phi.shape)==2
		res = self.TargetNetwork.predict(phi)
		return np.max(res)
	def predictQval(self,phi):
		assert len(phi.shape)==2
		return self.OnlineNetwork.predict(phi)

	def update(self):
		logging.info("update target network")
		self.TargetNetwork = copy.deepcopy(self.OnlineNetwork)

#TODO
class DQN:
	def __init__(self):
		pass

#TODO
class Dueling:
	def __init__(self):
		pass


