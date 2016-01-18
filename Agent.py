
'''
game agent
'''
from keras.models import Sequential,Graph
from keras.layers.core import Dense,Activation,Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Layer
from keras.utils import np_utils
from DQN import DDQN
from MemoryReplay import ReplayMemory
import random,time,copy
from ale_python_interface import ALEInterface
import logging
import argparse
import numpy as np
from random import randrange
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imshow,show
from defaults import defaults
class Agent:
	'''

	'''
	def __init__(self,para):
		model = self.__createNetwork()
		self.ale = ALEInterface()
		
		self.DDQN = DDQN(model,agentPara)
		self.RM = ReplayMemory(agentPara.memory_size,agentPara,gamePara)
		self.agentPara = agentPara
		self.gamePara = gamePara
		self.__setALE()
		self.legal_actions = self.ale.getLegalActionSet()
	

	#set ale's parameter
	def __setALE(self):
		
		self.ale.setInt('random_seed',3)
		self.ale.setBool('display_screen', True)
		self.ale.setInt('frame_skip',self.agentPara.frame_skip)
		self.ale.setInt('max_num_frames',self.agentPara.max_num_frames)
		self.ale.loadROM(self.agentPara.game)

	
	#create network using keras	
	def __createNetwork(self):
		model = Sequential()
		model.add(Convolution2D(32, 8, 8,border_mode='valid',input_shape=(4, 84, 84),subsample=(4,4)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 4, 4,subsample=(2,2)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 4, 4))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Dense(18))
		return model
	#action
	def __act(self,action):
		return self.ale.game_over(),self.ale.act(action)
	def __processFrame(self,frame):
		frame = np.asarray(frame,dtype=float)
		frame = resize(frame,(84,84))
		return frame
	
	#get current frame
	def __getFrame(self):
		frame = self.ale.getScreenRGB()
		return self.__processFrame(rgb2gray(frame))

	#epsilon decays
	def __eDecay(self):
		pass
		#TODO


	def __repeatAction(self,action):
		reward = 0
		state = []
		isTerminal = False
		for i in xrange(self.agentPara.action_repeat):
			if not isTerminal:
				isTerminal,r = self.__act(action)
				reward += r
			state.append(self.__getFrame())  
		state = self.__processState(state)
		return isTerminal,reward,state


	def __populateMemory(self):
		cntFrames = 0
		random.seed(time.time())
		action = self.legal_actions[randrange(len(self.legal_actions))]
		isTerminal,reward,phi = self.__repeatAction(action)
		while cntFrames < self.agentPara.replay_start_size:
			action = self.legal_actions[randrange(len(self.legal_actions))]
			#print action
			isTerminal,reward,phis = self.__repeatAction(action)
			while isTerminal:
				self.ale.reset_game()
				action = self.legal_actions[randrange(len(self.legal_actions))]
				isTerminal,reward,phi = self.__repeatAction(action)
				continue
			self.RM.store(phi,action,reward,phis)
			phi = copy.deepcopy(phis)
			cntFrames += 1


	def evaluate(self):
		pass
	def __processState(self,state):
		state = np.asarray(state)
		return state
	def __run_episode(self,epoch):
		print "now,epoch:",epoch
		
		cnt_frames = 0
		phis = None
		phi = None
		reward = 0
		action = 0
		while cnt_frames <= self.gamePara.max_frames:
			
			if cnt_frames <= self.agentPara.no_op_max:
				isTerminal,reward,phi = self.__repeatAction(0)
				print "no action"
				#print phi.max(axis=1)
				phis = copy.deepcopy(phi)
			else:
				#print phi.shape
				action = self.greedyAction(phi)
				print action
				isTerminal,reward,phis = self.__repeatAction(action)
			if isTerminal:
				self.ale.reset_game()
				action = self.legal_actions[randrange(len(self.legal_actions))]
				isTerminal,reward,phi = self.__repeatAction(action)
				continue
			self.RM.store(phi,action,reward,phis)
			cnt_frames += 1
			phi = copy.deepcopy(phis)


			#Sample training data
			Sphi,Sact,Srew,Sphis = self.RM.stochasticSample()
			#print Sphi.shape,Sphis.shape
			Sphi = np.asarray([Sphi[i] for i in xrange(self.agentPara.batch_size)])
			Sphis = np.asarray([Sphis[i] for i in xrange(self.agentPara.batch_size)])
			
			self.DDQN.train(Sphi,Sphis,Sact,Srew,self.agentPara)
			if cnt_frames % self.agentPara.rounds == 0:
				self.DDQN.update()
			#print self.DDQN.

		#TODO: figure


	#train my network by episodes
	def train(self):
		print "Training agent on "+self.gamePara.game
		print '-------------------------------------------------'
		print "Polulating replay memory"
		print '-------------------------------------------------'
		#self.__populateMemory()
		print "Done"

		print "Training"
		print '-------------------------------------------------'
		for epoch in xrange(self.agentPara.epoch_num):
			self.__run_episode(epoch)
			pass

	


	def greedyAction(self,phi):

		if len(self.RM.weights) < self.agentPara.replay_start_size:
			action = self.legal_actions[randrange(len(self.legal_actions))]
		else:
			#print np.asarray([phi]).shape
			pred = self.DDQN.predict(np.asarray([phi]))
			random.seed(time.time())
			if random.random() < self.agentPara.epislon:
				action = self.legal_actions[randrange(len(self.legal_actions))]
			else:
				#print "predict:",pred
				action = np.argmax(pred)
		return action

if __name__ == '__main__':
	#TODO:parser!!!
	gameParser = argparse.ArgumentParser()
	gameParser.add_argument("-epislon", type=float, default=0.1)
	gameParser.add_argument("-hist_len", type=int, default=4)
	gameParser.add_argument("-action_repeat", type=int, default=4)
	gameParser.add_argument("-learning_rate", type=float, default=0.00025)
	gameParser.add_argument("-momentum", type=float, default=0.95)
	gameParser.add_argument("-display_screen", type=bool, default=True)
	gameParser.add_argument("-frame_skip", type=int, default=1)
	gameParser.add_argument("-rounds", type=int, default=10000)
	gameParser.add_argument("-epoch_num", type=int, default=300)
	gameParser.add_argument("-max_frames", type=int, default=50000)
	gameParser.add_argument("-max_num_frames", type=int, default=1)
	gameParser.add_argument("-batch_size", type=int, default=64)
	gameParser.add_argument("-game",type=str,default='breakout.bin')
	gameParser.add_argument("-replay_start_size",type=int,default=50000)
	gameParser.add_argument("-memory_size",type=int,default=1000000)
	gameParser.add_argument("-gamma",type=float,default=0.99)
	gameParser.add_argument("-no_op_max",type=int,default=30)
	gameParser.add_argument("-e_init",type=float,default=1)
	gameParser.add_argument("-e_final",type=float,default=0.1)
	args = gameParser.parse_args()


	Ag = Agent(args)
	Ag.train()






