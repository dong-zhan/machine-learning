def d_sigmod(x):
	return x*(1-x)

def delta_rule(target_output, actual_output, weighted_sum, ith_input, learning_rate):
	return learning_rate * (target_output - actual_output) * d_sigmod(weighted_sum) * ith_input

class Node:
	def __init__(self):
		self.name = None			
		self.iw = []				#input weights
		self.inputs = []			#input nodes
		self.outputs = []			#output nodes
		
	def dump(self):
		print(self.name)
		print("inputs")
		for n in self.inputs:
			
		
			
	def fp(self):			#forward propagation	
		a = 0
		
	def bp(self):			#backpropagation
		a = 0
		
class Layer:
	def __init__(self):
		self.nodes = []
		b = 0
	
	def newNodes(self, n):
		for i in range(n):
			self.nodes.append(Node())
	
	def interconnect(self, layer):
		lenSrc = len(self.nodes)
		lenTarget = len(layer.nodes)
		for sn in self.nodes:
			for tn in layer.nodes:
