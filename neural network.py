import math

def sigmod(x):
	return 1/(1+math.exp(-x))
	
def d_sigmod(x):
	return x*(1-x)

def delta_rule(target_output, actual_output, ith_input):
	#ds = d_sigmod(actual_output)
	#print(target_output, actual_output, actual_output - target_output, ds, ith_input)
	return (actual_output - target_output) * d_sigmod(actual_output) * ith_input

class Node:
	def __init__(self, name = None):
		self.name = name	
		self.oIdx = []				#save the index in target node
		self.iw = []				#input weights
		self.inputs = []			#input nodes
		self.outputs = []			#output nodes
		self.ov = 0					#output value: activation_function(weighted_sum + b)
		self.dEdx = 0				#derivative error w.r.t. total inputs x
		self.dEdy = 0				#derivative error w.r.t. output y
		
	def dot_iw_inputs(self):		#computes weighted_sum
		lw = len(self.iw)
		d = 0
		for i in range(lw):
			d += self.iw[i] * self.inputs[i].ov
			#print(self.iw[i], self.inputs[i].ov)
		return d
				
	def dump(self):
		print("node name = ", self.name, "output value = ", self.ov)
		print("inputs")
		li = len(self.inputs)
		for i in range(li):
			print(self.inputs[i].name, self.iw[i])
		lo = len(self.outputs)
		print("outputs")
		for i in range(lo):
			print(self.outputs[i].name)
			
	def connect(self, tn, w):			#tn: target node, w: weight
		lenTnIw = len(tn.iw)
		tn.iw.append(w)
		tn.inputs.append(self)
		self.outputs.append(tn)
		self.oIdx.append(lenTnIw)
		
	def dump_output(self):
		lo = len(self.outputs)
		for i in range(lo):
			node = self.outputs[i]
			idx = self.oIdx[i]
			w = node.iw[idx]
			print(node.name, w)
		
	def compute_error(self, tv):		#tv: target value
		d = tv - self.ov
		return 0.5 * d * d 
			
	def fp(self, b):		#forward propagation, b: bias
		self.ov = sigmod(self.dot_iw_inputs() + b)
		
	def get_d_ith_weight(self, target_output, i):		
		return delta_rule(target_output, self.ov, self.inputs[i].ov)
		
	def bp1(self, target_output, i, learning_rate):			#backpropagation
		dw = self.get_d_ith_weight(target_output, i)
		self.iw[i] -= learning_rate * dw
		print("bp dump", self.name, i, self.iw[i])
		
	def bp(self, target_output, i, learning_rate):			#backpropagation
		self.dEdy = self.ov - target_output
		self.dEdx = self.dEdy * d_sigmod(self.ov)
		self.iw[i] -= learning_rate * self.dEdx * self.inputs[i].ov
		#print("bp dump", self.name, i, self.iw[i])
		
	def compute_dEdy(self):
		#sum all errors come into this node(backward direction)
		lo = len(self.outputs)
		dEdy = 0
		for i in range(lo) :
			node = self.outputs[i]
			idx = self.oIdx[i]
			dEdy += node.iw[idx] * node.dEdx
		return dEdy
		
	def bph(self, i, learning_rate):			#backpropagation for hidden layers
		self.dEdy = self.compute_dEdy()
		self.dEdx = self.dEdy * d_sigmod(self.ov)
		self.iw[i] -= learning_rate * self.dEdx * self.inputs[i].ov
		#print("bp dump", self.name, i, self.iw[i])		

class Layer:
	def __init__(self, name = None, initB = 0, prev = None, next = None):
		self.prev = prev
		self.next = next
		self.name = name
		self.nodes = []
		self.b = initB
		
	def connect(self, layer):
		self.next = layer
		layer.prev = self
		
	def fp(self):			#forward propagation	
		lenNodes = len(self.nodes)	
		for n in self.nodes:
			n.fp(self.prev.b)
			
	def bp(self):			#backpropagation
		pass
		
	def dump(self):
		lenNodes = len(self.nodes)
		print("Layer name =", self.name, "total nodes = ", lenNodes, "b = ", self.b)
		for n in self.nodes:
			n.dump()
	
	def newNodes(self, n):
		for i in range(n):
			self.nodes.append(Node())
