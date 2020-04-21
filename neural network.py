import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import random

def sigmoid(x):
	#return math.tanh(x)
	return 1/(1+math.exp(-x))
	
def d_sigmoid(x):
	#return 1 - x*x
	return x*(1-x)

def delta_rule(target_output, actual_output, ith_input):
	#ds = d_sigmoid(actual_output)
	#print(target_output, actual_output, actual_output - target_output, ds, ith_input)
	return (actual_output - target_output) * d_sigmoid(actual_output) * ith_input

	

###############################################################################
##						neural network node
###############################################################################
class Node:
	def __init__(self, name = None):
		self.name = name	
		self.oIdx = []				#index of current node in the target node's input nodes
		self.iw = []				#input weights
		self.inputs = []			#input nodes
		self.outputs = []			#output nodes
		self.ov = 0					#output value: activation_function(weighted_sum + b)
		self.dEdx = 0				#error derivative w.r.t. total inputs x
		self.dEdy = 0				#error derivative w.r.t. output y
		self.bias = 0

	def reset(self):
		lw = len(self.iw)
		r = math.sqrt(6/lw)
		for i in range(lw):
			self.iw[i] = r * np.random.normal(1, 0.2)		#loc: 1, deviation: 0.2
			if random.uniform(-1, 1) > 0 :
				self.iw[i] *= -1
			
		self.bias = 0

	def getFormula(self):
		text = ""
		i = 0
		lw = len(self.iw)
		txt = "{:.5f}"
		for w in self.iw:
			text += txt.format(w)
			text += "x"
			text += str(i)
			i = i + 1
			if i != lw :
				text += " + "

		text += " + "
		text += txt.format(self.bias)
		return text
		
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
			
	def connect(self, tn, w = 0):			#tn: target node, w: weight
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
			
	def fp(self):		#forward propagation
		self.ov = sigmoid(self.dot_iw_inputs() + self.bias)
		
	def get_d_ith_weight(self, target_output, i):		
		return delta_rule(target_output, self.ov, self.inputs[i].ov)
		
	def bp1(self, target_output, i, learning_rate):			#backpropagation
		dw = self.get_d_ith_weight(target_output, i)
		self.iw[i] -= learning_rate * dw
		#print("bp dump", self.name, i, self.iw[i])

	def bp(self, target_output, learning_rate):			#backpropagation, target_output is example
		self.dEdy = self.ov - target_output
		self.dEdx = self.dEdy * d_sigmoid(self.ov)
		tmp = learning_rate * self.dEdx
		lenw = len(self.iw)
		#old = self.iw[0]
		for i in range(lenw):
			self.iw[i] -= tmp * self.inputs[i].ov

		self.bias -= tmp
		#print(old, "-->", self.iw[0])
		
	def compute_dEdy(self):
		#sum all derrors that come into this node(backward direction)
		lo = len(self.outputs)
		dEdy = 0
		for i in range(lo) :
			node = self.outputs[i]
			idx = self.oIdx[i]
			dEdy += node.iw[idx] * node.dEdx
		return dEdy
		
	def bph(self, learning_rate):			#backpropagation for hidden layers
		self.dEdy = self.compute_dEdy()
		self.dEdx = self.dEdy * d_sigmoid(self.ov)
		tmp = learning_rate * self.dEdx
		lenw = len(self.iw)
		for i in range(lenw):
			self.iw[i] -= tmp * self.inputs[i].ov

		self.bias -= tmp
		#print("bp dump", self.name, self.bias)		

###############################################################################
##						neural network
###############################################################################
class NNetwork:
	def __init__(self, learning_rate, epochs):
		self.learning_rate = learning_rate
		self.first_layer = None
		self.last_layer = None
		self.epochs = epochs
		self.logInterval = 0
		self.logFirst20 = False

	def reset(self):
		layer = self.first_layer.next
		while layer:
			layer.reset()
			layer = layer.next

	def set_first_layer(self, first_layer):
		self.first_layer = first_layer

		#find last_layer
		layer = first_layer.next
		while layer:
			self.last_layer = layer
			layer = layer.next

	def fp(self, *args):		#args: feachers
		self.first_layer.initLayerForTrain1(*args)
		
		#forward propagation
		layer = self.first_layer.next
		while layer:
			layer.fp()
			layer = layer.next

		#print(self.first_layer.nodes[0].ov, self.first_layer.nodes[1].ov, "--->", self.last_layer.nodes[0].ov)

		return self.last_layer.nodes[0].ov

	def dump_formula(self):
		layer = self.first_layer.next
		while layer:
			layer.dump_formula()
			layer = layer.next
		print("\n")

	def train(self, *args):		#args: first is label, features rest
		argCount = len(args)

		li = 0
	
		nExamples = len(args[0])

		for e in range(self.epochs):

			#choose an example
			iexample = random.randrange(nExamples)

			#init layer0		#TODO move the function here to avoid packing/unpacking
			self.first_layer.initLayerForTrain(iexample, *args)	

			#forward propagation
			layer = self.first_layer.next
			while layer:
				layer.fp()
				layer = layer.next
			
			#first value should always be 0.5, if sigmoid is the activation function
			#print(layer0.nodes[0].ov, layer0.nodes[1].ov, "-fp->", self.last_layer.nodes[0].ov, "example:", args[0][iexample])

			#backward propagation
			layer = self.last_layer
			layer.bp(args[0][iexample])
			layer = layer.prev
			while layer:
				if layer == self.first_layer:
					break
				layer.bph()
				layer = layer.prev

			if self.logInterval > 0:
				li += 1
				if self.logFirst20:
					self.dump_formula()	
					if li > 20:
						self.logFirst20 = False
				else:
					if li > self.logInterval :
						li = 0
						self.dump_formula()			#this can be used to observe convergency


###############################################################################
##						neural network layer
###############################################################################
class Layer:
	def __init__(self, nnetwork, layerName, autoName, nNodes):
		self.prev = None
		self.next = None
		self.nnetwork = nnetwork
		self.nodes = []
		self.name = layerName

		if nNodes > 0:
			self.newNodes(nNodes)

		if autoName:
			self.autoNameNodes()
				
	def dump_formula(self):
		for node in self.nodes:
			print(node.getFormula())
						
	def connectAllNodes(self, nextLayer):
		for n1 in self.nodes :
			for n2 in nextLayer.nodes:
				n1.connect(n2)
					
	def connect(self, nextLayer, autoConnectNodes = None):
		self.next = nextLayer
		nextLayer.prev = self
		
		if autoConnectNodes:
			self.connectAllNodes(nextLayer)
		
	def fp(self):			#forward propagation	
		lenNodes = len(self.nodes)	
		for n in self.nodes:
			n.fp()
			
	def bp(self, target_output):			#backpropagation
		lenNodes = len(self.nodes)	
		for n in self.nodes:
			n.bp(target_output, self.nnetwork.learning_rate)

	def bph(self):			#backpropagation for hidden layers
		lenNodes = len(self.nodes)	
		for n in self.nodes:
			n.bph(self.nnetwork.learning_rate)
			
	def dump(self):
		lenNodes = len(self.nodes)
		print("Layer name =", self.name, "total nodes = ", lenNodes)
		for n in self.nodes:
			n.dump()
	
	def newNodes(self, n):
		for i in range(n):
			self.nodes.append(Node())

	def autoNameNodes(self):
		ln = len(self.nodes)
		for i in range(ln):
			self.nodes[i].name = self.name + "_node" + str(i)
		
	def reset(self):
		for node in self.nodes:
			node.reset()

	def initLayerForTrain1(self, *args):		#args: features
		ln = len(self.nodes)
		for inode in range(ln):
			self.nodes[inode].ov = args[inode]

	def initLayerForTrain(self, iExample, *args):		#args: first label, features rest
		ln = len(self.nodes)
		for inode in range(ln):
			#print(args[0])			#no idea how args is packed... confusing, sometimes iArg is packed too, sometimes it's not...
			self.nodes[inode].ov = args[inode+1][iExample]
			#print(self.nodes[inode].ov)




