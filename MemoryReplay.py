'''
memory replay
'''
import pickle as pkle
import os
import numpy as np
import h5py
class ReplayMemory():
	def __init__(self,memory_size,rng):
		self.f = h5py.File('../drl_memory/memory.hdf5', 'a')
		self.curSz = 0 
		self.MSize = memory_size
		self.rng = rng
	def __setMemory(self,idx,phi,action,reward,phis,terminal):
		#print phi.shape
		self.f['phi'][idx] = phi
		self.f['action'][idx] = action
		self.f['reward'][idx] = reward
		self.f['phis'][idx] = phis
		self.f['terminal'][idx] = terminal
	def add_sample(self,phi,action,reward,phis,terminal):
		if self.curSz < self.MSize:
			self.__setMemory(self.curSz,phi,action,reward,phis,terminal)
			self.curSz+=1
		else:
			idx = self.rng.randint(0,self.MSize)
			self.__setMemory(idx,phi,action,reward,phis,terminal)
	def stochasticSample(self,batch_size):
		expIdx = self.rng.choice(range(self.curSz),size=batch_size,replace=False)
		phi = np.asarray([self.f['phi'][idx] for idx in expIdx])
		action = np.asarray([self.f['action'][idx] for idx in expIdx])
		reward = np.asarray([self.f['reward'][idx] for idx in expIdx])
		phis = np.asarray([self.f['phis'][idx] for idx in expIdx])
		terminal = np.asarray([self.f['terminal'][idx] for idx in expIdx])
		return phi,action,reward,phis,terminal	
