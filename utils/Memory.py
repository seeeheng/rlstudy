class Memory:
	def __init__(self):
		self.memory = [] 

	def add_into_memory(self, stuff):
		self.memory += [stuff]

	def get_and_clear_memory(self):
		# This function gets called at the end of each episode, when the monte-carlo agent is learning from the episodes.
		memory = self.memory
		self.memory = []
		return memory