import logging

class TrainMyAgent:

	def __init__(self,args,ale,agent,valid_actions,rng):
		self.valid_actions = valid_actions
		self.num_valid_action = len(valid_actions)
		self.rng = rng
		self.ale = ale
		self.agent = agent
		self.frame_height,self.frame_width = ale.getScrrenDims()
		self.resized_frame_height,self.resized_frame_width = args.resized_frame_height,args.resized_frame_width
		self.num_epoch = args.num_epoch
		self.steps_per_epoch = args.steps_per_epoch
		self.steps_per_test = args.steps_per_test
		
		self.action_repeat = args.action_repeat
		self.no_op_max = args.no_op_max
		self.batch_size = args.batch_size
		self.update_requency = args.update_requency
		self.target_nn_update_frequency = args.target_nn_update_frequency
		

	def run(self):
		for _ in xrange(self.num_epoch):
			self.run_epoch(_)
			self.agent.finish_epoch(_,self.steps_per_epoch,True)
			if self.steps_per_test > 0:
				self.agent.start_testing()
				self.run_epoch(_,self.steps_per_test,False)
				self.agent.finish_test(_)



	def run_epoch(self,epoch,num_steps,trainable):
		num_steps_left = num_steps

		while num_steps_left > 0:
			t = "training" if trainable else "testing"
			info = t+ " on epoch "+str(epoch)+" and "+str(num_steps_left)+" steps left"
			logging.info(info)
			self.run_episode(num_steps_left,trainable)

		pass

	def __init_episode(self):
		
		pass

	def __act(self,action):
		reward = self.ale.act(action)
		#TODO save current frame into buffer
		return reward
	def __step(self):
		reward = 0
		for _ in xrange(self.action_repeat):
			reward += self.__act(action)
		
		return reward

	def run_episode(self,num_steps_left,trainable):
		self.__init_episode()
		start_lives = self.ale.lives()
		action = self.agent.start_episode()
		step_cnt = 0
		while True:
			reward = self.__step(action)



	def get_frame(self):
		pass


	def resize_frame(self):
		pass