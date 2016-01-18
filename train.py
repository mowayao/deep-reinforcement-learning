from ale_python_interface import ALEInterface

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
		self.replay_start_size = args.replay_start_size
		self.action_repeat = args.action_repeat
		self.no_op_max = args.no_op_max
		self.batch_size = args.batch_size
		self.update_requency = args.update_requency
		self.target_nn_update_frequency = args.target_nn_update_frequency
		

	def run(self):
		pass



	def run_epoch(self):

		pass

	def __init_episode(self):

		pass

	def __act(self):
		pass


	def __step(self):
		pass

	def run_episode(self):
		pass


	def get_frame(self):
		pass


	def resize_frame(self):
		pass