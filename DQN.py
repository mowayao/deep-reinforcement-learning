
import copy
import numpy as np
'''
double deep Q-learning network
based on keras,a python-based deep neural network framework
'''

class DDQN:
	def __init__(self,network,valid_actions):
		self.OnlineNetwork = network
		self.TargetNetwork = network
		self.valid_actions = valid_actions
		self.num_valid_actions = len(valid_actions)
	def choose_action(self,phi):
		res = self.predict(phi)
		return self.valid_actions[np.argmax(res)]

	def train(self,phi1,phi2,act,reward,terminals):


		pass
		#return loss
		#pre1 = self.OnlineNetwork.predict(phi1)
		#pre2 = self.OnlineNetwork.predict(phi2)
		#bestAction = pre2.argmax(axis=1)
		#pre3 = self.TargetNetwork.predict(phi2)
		#pre1[:,bestAction] = reward+pre3[:,bestAction]*para.gamma
		
		#TODO:consider terminal
		#self.OnlineNetwork.train_on_batch(phi1,pre1)
		#print "loss:",self.OnlineNetwork.evaluate(phi1,pre1)
		#print "average Q value:",np.mean(pre1)
	def predict(self,phi):
		return self.TargetNetwork.predict(phi)

	def update(self):
		print "update target"
		self.TargetNetwork = copy.deepcopy(self.OnlineNetwork)

#TODO
class DQN:
	def __init__(self):
		pass

#TODO
class Dueling:
	def __init__(self):
		pass


