
from keras.models import Sequential,Graph
from keras.layers.core import Dense,Activation,Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Layer
from keras.optimizers import RMSprop
from keras.utils import np_utils
import argparse
from DQN import DDQN
from defaults import defaults
from ale_python_interface import ALEInterface
from MemoryReplay import ReplayMemory
import numpy as np 
import logging
import os
from Agent import Agent
from train import TrainMyAgent
def getParameters():
	myParser = argparse.ArgumentParser()
	myParser.add_argument("-epsilon_start", type=float, default=defaults.epsilon_start)
	myParser.add_argument("-epsilon_min",type = float,default=defaults.epsilon_min)

	myParser.add_argument("-phi_len", type=int, default=defaults.phi_len)
	myParser.add_argument("-action_repeat", type=int, default=defaults.action_repeat)
	myParser.add_argument("-resized_height",type=int,default=defaults.resized_height)
	myParser.add_argument("-resized_width",type=int,default=defaults.resized_width)
	myParser.add_argument("-learning_rate", type=float, default=defaults.learning_rate)
	myParser.add_argument("-rmsp_rho", type=float, default=defaults.rmsp_rho)
	myParser.add_argument("-rmsp_epsilon",type=float,default=defaults.rmsp_epsilon)
	myParser.add_argument("-display_screen", type=bool, default=defaults.display_screen)
	myParser.add_argument("-frame_skip", type=int, default=defaults.frame_skip)
	myParser.add_argument("-epsilon_decay", type=int, default=defaults.epsilon_decay)
	myParser.add_argument("-num_epoch", type=int, default=defaults.epoch_num)
	myParser.add_argument("-steps_per_test", type=int, default=defaults.steps_per_test)
	myParser.add_argument("-steps_per_epoch", type=int, default=defaults.steps_per_epoch)
	myParser.add_argument("-batch_size", type=int, default=defaults.batch_size)
	myParser.add_argument("-game",type=str,default=defaults.rom)
	myParser.add_argument("-base_rom_path",type=str,default=defaults.base_rom_path)
	myParser.add_argument("-replay_start_size",type=int,default=defaults.replay_start_size)
	myParser.add_argument("-memory_size",type=int,default=defaults.memory_size)
	myParser.add_argument("-discount",type=float,default=defaults.discount)
	myParser.add_argument("-no_op_max",type=int,default=defaults.no_op_max)
	myParser.add_argument("-update_frequency",type=int,default=defaults.update_frequency)
	myParser.add_argument("-target_nn_update_frequency",type=int,default=defaults.target_nn_update_frequency)
	myParser.add_argument("-repeat_action_probability",type=float,default=defaults.repeat_action_probability)
	myParser.add_argument("-death_end_episode",type=bool,default=defaults.death_end_episode)
	myParser.add_argument("-buffer_size",type=int,default=defaults.buffer_size)
	myParser.add_argument("-exps_prefix",type=str,default=defaults.exps_prefix)
	myParser.add_argument("-holdout_data_size",type=int,default=defaults.holdout_data_size)
	args = myParser.parse_args()

	return args

def buildNetwork(height,width,rmsp_epsilon,rmsp_rho,learning_rate,num_valid_action):
	model = Sequential()
	model.add(Convolution2D(32, 8, 8,border_mode='valid',input_shape=(4, height,width),subsample=(4,4)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 4, 4,subsample=(2,2)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 4, 4))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Dense(num_valid_action))
	rmsp = RMSprop(lr=learning_rate,rho=rmsp_rho,epsilon=rmsp_epsilon)
	logging.info("compiling network model")
	model.compile(loss='mse',optimizer=rmsp)
	logging.info("compilation complete")
	return model


def launch():
	logging.basicConfig(level=logging.INFO)
	myArgs = getParameters()
	rom = myArgs.game
	full_rom_path = os.path.join(myArgs.base_rom_path,rom)
	rng = np.random.RandomState()
	ale = ALEInterface()
	ale.setInt('random_seed',38)
	ale.setBool('display_screen',myArgs.display_screen)
	ale.setInt('frame_skip',myArgs.frame_skip)
	ale.setFloat('repeat_action_probability',myArgs.repeat_action_probability)

	ale.loadROM(full_rom_path)
	valid_actions = ale.getMinimalActionSet()
	'''for episode in xrange(10):
		total_reward = 0
		while not ale.game_over():
			from random import randrange
			a = valid_actions[randrange(len(valid_actions))]
			ale.act(a)
			#print reward
			#print ale.getScreenRGB()

			#total_reward += reward
			#print 'Episode', episode, 'ended with score:', total_reward
		ale.reset_game()
	'''
	memory_pool = ReplayMemory(myArgs.memory_size,rng)
	network_model = buildNetwork(myArgs.resized_height,myArgs.resized_width,myArgs.rmsp_epsilon,myArgs.rmsp_rho,myArgs.learning_rate,len(valid_actions))
	ddqn = DDQN(network_model,valid_actions,myArgs.target_nn_update_frequency,myArgs.discount)
	agent = Agent(myArgs,ddqn,memory_pool,valid_actions,rng)
	train_agent = TrainMyAgent(myArgs,ale,agent,valid_actions,rng)
	train_agent.run()

if __name__=="__main__":
	launch()