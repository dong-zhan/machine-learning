import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import random

def sigmoid(x):
	return x
	return 1/(1+math.exp(-x))
	
def d_sigmoid(x):
	return 1
	return x*(1-x)

def delta_rule(target_output, actual_output, ith_input):
	#ds = d_sigmoid(actual_output)
	#print(target_output, actual_output, actual_output - target_output, ds, ith_input)
	return (actual_output - target_output) * d_sigmoid(actual_output) * ith_input

###############################################################################
#LineFeatureLabel holds one feature and one label which are close to a line
###############################################################################
class LineFeatureLabel:
	def __init__(self):
		self.X = None			#feature
		self.Y = None			#label
		self.Side = None			

	def dump(self):
		print(self.X)
		print(self.Y)

	def get_random_point_along_line_2d(self, startPos, dir, t, noise, cross):
		pos = vector2()
		pos = startPos + dir * t
		side = random.uniform(-1, 1)
		pos += cross * side * noise		#TODO: this uniform() has problems???
		return pos, side

	def generate_line_set_2d(self, startPos, stopPos, n, noise):		
		dir = stopPos - startPos
		lenDir = dir.length()
		step = lenDir / n
		dir /= lenDir
		cross = vector2(dir.y, -dir.x)
		self.X = np.zeros(n)
		self.Y = np.zeros(n)
		self.Side = np.zeros(n)
		pos = vector2()
		for i in range(n):
			pos = startPos + dir * i * step
			pos, side = self.get_random_point_along_line_2d(startPos, dir, i * step, noise, cross)
			self.X[i] = pos.x
			self.Y[i] = pos.y
			self.Side[i] = side

def test():
	global feature
	feature = LineFeatureLabel()
	feature.generate_line_set_2d(vector2(-1,0), vector2(1,0), 10, 0.01)
	print(feature.X, feature.Y)

	
###############################################################################
#LineFeatureLabel holds 2 features and one label which are close to a line
#because it's for test only, so...
###############################################################################
class LineFeatureLabel2:
	def __init__(self, p0, p1, n, noise):
		self.Y = np.zeros(n)	#label

		self.p0 = p0
		self.p1 = p1

		lfa = LineFeatureLabel()
		dir1 = p1 - p0
		lfa.generate_line_set_2d(p0, p1, n, noise)

		#TODO: make this random
		for i in range(n):
			self.Y[i] = lfa.Side[i]

		self.X1 = lfa.X
		self.X2 = lfa.Y

	def dump(self):
		ly = len(self.Y)
		for i in range(ly):
			print(self.X1[i], self.X2[i], self.Y[i])


def test():
	global feature
	feature = LineFeatureLabel2(vector2(-1,0), vector2(1,0), 20, 0.1)
	feature.dump()

###############################################################################
##						neural network node
###############################################################################
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

	def reset(self):
		for w in self.iw:
			w = 0
		
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
		self.ov = sigmoid(self.dot_iw_inputs() + b)
		
	def get_d_ith_weight(self, target_output, i):		
		return delta_rule(target_output, self.ov, self.inputs[i].ov)
		
	def bp1(self, target_output, i, learning_rate):			#backpropagation
		dw = self.get_d_ith_weight(target_output, i)
		self.iw[i] -= learning_rate * dw
		#print("bp dump", self.name, i, self.iw[i])

	def bp(self, target_output, learning_rate):			#backpropagation
		self.dEdy = self.ov - target_output
		self.dEdx = self.dEdy * d_sigmoid(self.ov)
		tmp = learning_rate * self.dEdx
		lenw = len(self.iw)
		#old = self.iw[0]
		for i in range(lenw):
			self.iw[i] -= tmp * self.inputs[i].ov
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
		#print("bp dump", self.name, i, self.iw[i])		

###############################################################################
##						neural network
###############################################################################
class NNetwork:
	def __init__(self, learning_rate, epochs):
		self.learning_rate = learning_rate
		self.first_layer = None
		self.last_layer = None
		self.epochs = epochs

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

		return self.last_layer.nodes[0].ov

	def train(self, *args):		#args: first is label, features rest
		argCount = len(args)
	
		nExamples = len(args[0])

		for e in range(self.epochs):

			#choose an example
			iexample = random.randrange(nExamples)

			#init layer0		#TODO move the function here to avoid packing/unpacking
			#print("init layer0")
			self.first_layer.initLayerForTrain(iexample, *args)	

			#forward propagation
			layer = self.first_layer.next
			while layer:
				layer.fp()
				layer = layer.next
			
			#print(self.first_layer.nodes[0].ov)
			#print(self.last_layer.nodes[0].ov)

			#backward propagation
			layer = self.last_layer
			layer.bp(args[0][iexample])
			layer = layer.prev
			while layer:
				if layer == self.first_layer:
					break
				layer.bph()
				layer = layer.prev

			#print(self.first_layer.nodes[0].ov, self.last_layer.nodes[0].ov, args[0][iexample], self.last_layer.nodes[0].iw)


###############################################################################
##						neural network layer
###############################################################################
class Layer:
	def __init__(self, nnetwork = None, name = None, initB = 0, prev = None, next = None):
		self.prev = prev
		self.next = next
		self.name = name
		self.nodes = []
		self.b = initB
		self.nnetwork = nnetwork
		
	def connect(self, layer):
		self.next = layer
		layer.prev = self
		
	def fp(self):			#forward propagation	
		lenNodes = len(self.nodes)	
		for n in self.nodes:
			n.fp(self.prev.b)
			
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
		print("Layer name =", self.name, "total nodes = ", lenNodes, "b = ", self.b)
		for n in self.nodes:
			n.dump()
	
	def newNodes(self, n):
		for i in range(n):
			self.nodes.append(Node())

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


#plot_sigmoid is an idea
def plot_sigmoid(x0, x1, n, m, b) :
	step = (x1-x0)/n
	X = np.zeros(n)
	Y = np.zeros(n)
	for i in range(n):
		X[i] = x0 + i * step
		Y[i] = sigmoid( X[i] * m + b)

	plt.axis([-0.1, 1, -1, 1])
	plt.plot(X, Y)		#x,y both list []
	plt.show()

plot_sigmoid(0, 1, 20, 0.707, 0)


#x axis is inferred from listX
def scatter_2d(x0, x1, y0, y1, listX, listY):
	plt.axis([x0, x1, y0, y1])
	plt.scatter(listX, listY)		#x,y both list []
	plt.show()


