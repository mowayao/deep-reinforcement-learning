import logging
import numpy as np
class TrainMyAgent:

	def __init__(self,args,ale,agent,valid_actions,rng):
		self.valid_actions = valid_actions
		self.num_valid_action = len(valid_actions)
		self.rng = rng
		self.ale = ale
		self.agent = agent
		self.frame_height,self.frame_width = ale.getScreenDims() #####WTF!!!!!!!! BUG from ALE!!!!!!!
		self.resized_frame_height,self.resized_frame_width = args.resized_height,args.resized_width
		self.num_epoch = args.num_epoch
		self.steps_per_epoch = args.steps_per_epoch
		self.steps_per_test = args.steps_per_test
		self.death_end_episode = args.death_end_episode
		self.buffer_size = args.buffer_size
		self.action_repeat = args.action_repeat
		self.no_op_max = args.no_op_max
		self.batch_size = args.batch_size
		self.update_requency = args.update_frequency
		self.target_nn_update_frequency = args.target_nn_update_frequency
		self.flag = False
		self.frame_buffer = np.empty((self.buffer_size,self.frame_width,self.frame_height),dtype=np.uint8)
		self.frame_buffer_cnt = 0
	def run(self):
		for _ in xrange(self.num_epoch):
			self.run_epoch(_,self.steps_per_epoch,True)
			self.agent.finish_epoch(_)
			if self.steps_per_test > 0:
				self.agent.start_testing()
				self.run_epoch(_,self.steps_per_test,False)
				self.agent.finish_test(_)



	def run_epoch(self,epoch,num_steps,trainable):
		num_steps_left = num_steps
		self.trainable = trainable
		while num_steps_left > 0:
			t = "training" if trainable else "testing"
			info = t+ " on epoch "+str(epoch)+" and "+str(num_steps_left)+" steps left"
			logging.info(info)
			_,steps = self.run_episode(num_steps_left,trainable)
			num_steps_left -= steps


	def __init_episode(self):
		'''
		perform no operation to initial the episode
		'''
		if not self.flag or self.ale.game_over():
			self.ale.reset_game()
			if self.no_op_max > 0:
				num_random_actions = self.rng.randint(0,self.no_op_max)+5
				for _ in xrange(num_random_actions):
					self.__act(0)
				for _ in xrange(4-num_random_actions%4):
					self.__act(0)
	def __act(self,action):
		reward = self.ale.act(action)
		self.frame_buffer_cnt %= 200
		self.ale.getScreenGrayscale(self.frame_buffer[self.frame_buffer_cnt])
		self.frame_buffer_cnt += 1
		return reward
	def __step(self,action):
		reward = 0
		for _ in xrange(self.action_repeat):
			reward += self.__act(action)
		return reward
	def __get_phi(self):
		from skimage.transform import resize
		phi = [resize(self.frame_buffer[_],(self.resized_frame_height,self.resized_frame_width)) for _ in xrange(self.frame_buffer_cnt-4,self.frame_buffer_cnt)]
		return np.asarray(phi)
	def run_episode(self,num_steps_left,trainable):
		self.__init_episode()
		start_lives = self.ale.lives()
		action = self.agent.start_episode(self.__get_phi())
		step_cnt = 0
		while True:
			reward = self.__step(action)
			self.flag = self.ale.lives() < start_lives  and  self.death_end_episode
			terminal = self.ale.game_over() or self.flag
			step_cnt += 1
			if terminal or step_cnt > num_steps_left:
				self.agent.finish_episode(reward,terminal,trainable)
				break
			action = self.agent.step(reward,self.__get_phi(),trainable)
		return terminal,step_cnt
