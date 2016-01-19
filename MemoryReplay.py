'''
memory replay
'''
import pickle as pkle
import os
import numpy as np
import h5py
class ReplayMemory():
	def __init__(self,memory_size,rng):
		self.f = h5py.File('../memory.hdf5', 'a')
		self.curSz = 0 
		self.MSize = memory_size
	def __setMemory(self,idx,phi,action,reward,phis,terminal):
		#print phi.shape
		self.f['phi'][idx] = phi
		self.f['action'][idx] = action
		self.f['reward'][idx] = reward
		self.f['phis'][idx] = phis
		self.f['terminal'][idx] = terminal
	def store(self,phi,action,reward,phis,terminal):
		if self.curSz < self.MSize:
			self.__setMemory(self.curSz,phi,action,reward,terminal)
			self.curSz+=1
		else:
			idx = rng.randint(0,self.MSize)
			self.__setMemory(idx,phi,action,reward,phis,terminal)
	def stochasticSample(self,batch_size):
		expIdx = np.random.choice(range(self.curSz),size=batch_size,replace=False)
		phi = [self.f['phi'][idx] for idx in expIdx]
		action = [self.f['action'][idx] for idx in expIdx]
		reward = [self.f['reward'][idx] for idx in expIdx]
		phis = [self.f['phis'][idx] for idx in expIdx]
		terminal = [self.f['terminal'][idx] for idx in expIdx]
		return phi,action,reward,phis,terminal
