# color of the canvas
# 1, get predicated value for all examples
# 2, compare the predicated with example value, put the difference in each pixel
# 3, find all known differences for current pixel in a neighborhood
# 4, weighted average as current pixel value, use distance as weight?

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

###############################################################################
##							LineForCanvas
###############################################################################
## for plotting a line on a bitmap in matplotlib
class LineForCanvas:
	def __init__(self, m, b, epsilon):	 
		self.m = m
		self.b = b
		self.Z = None
		self.fillColor = 0
		self.epsilon = epsilon
		self.vFillFunc = np.vectorize(self.fillFunc, excluded='self')

	def fillFunc(self, x, y):
		#v = sigmoid(x)
		v = self.m * x + self.b
		if abs(v - y) < self.epsilon :
		#if v-y > 0:			#this splits the space into 2 halves with different colors.
			return self.fillColor
		else:
			return 0

	def fill(self, X, Y, color) :
		self.fillColor = color
		self.Z = self.vFillFunc(X, Y)


def test():
	global mc
	#mc = Canvas(-6, 6, 0.01, -1, 1, 0.01)			#test sigmoid
	mc = Canvas(-1, 1, 0.01, -1, 1, 0.01)	
	lc = LineForCanvas(1, 0, 0.01)
	lc.fill(mc.X, mc.Y, 20)
	mc.Z = lc.Z
	mc.plot()


###############################################################################
##							MCanvas
###############################################################################
class MCanvas(Canvas) : 
	def __init__(self, x0, x1, xstep, y0, y1, ystep):
		super().__init__(x0, x1, xstep, y0, y1, ystep)

	def draw_feature(self, X, Y, Z):
		lx = len(X)
		for i in range(lx):
			color = -1
			if Z[i] > 0:
				color = 1
			self.setColor(X[i], Y[i], color)

###############################################################################
##							NNetworkForCanvas
###############################################################################
##use this on trained network
## for plotting nerual network on a bitmap in matplotlib
class NNetworkForCanvas(Canvas) :
	def __init__(self, network):   
		self.network = network
		self.Z = None
		self.fillColor = 0
		self.vLayerFunc = np.vectorize(self.layerFunc, excluded='self')

	def layerFunc(self, x, y):
		v = self.network.fp(x, y)
		#print(x, y, v)
		if v > 0.5 :		#sigmoid's middle point is 0.5, if it's linear, then, this should be 0
			return 0.5
		return -0.5

	def fill(self, X, Y, color) :
		self.fillColor = color
		self.Z = self.vLayerFunc(X, Y) 





###############################################################################
##							create_4_feature_layers
###############################################################################
def test():
	create_4_feature_layers()	

	global mc
	mc = MCanvas(-1, 1, 0.01, -1, 1, 0.01)		# 2/0.01 pixels
	lc = NNetworkForCanvas(nnetwork)
	lc.fill(mc.X, mc.Y, 20)
	mc.Z = lc.Z

	mc.draw_feature(sphere.X1, sphere.X2, sphere.Y)

	mc.plot()

def create_4_feature_layers():
	global layer0, layer1, layer2, layer3, feature, nnetwork, sphere

	total_layers = 3

	sphere = SphereFeatureLabel()

	if total_layers == 4 :
		nnetwork = NNetwork(0.001, 20000)
	
		layer0 = Layer(nnetwork, "layer0", True, 2)
		layer1 = Layer(nnetwork, "layer1", True, 4)
		layer2 = Layer(nnetwork, "layer2", True, 2)	
		layer3 = Layer(nnetwork, "layer3", True, 1)	

		layer0.connect(layer1, True)
		layer1.connect(layer2, True)
		layer2.connect(layer3, True)

		v = 1
		sphere.addInsideSphere(vector2(0,0), 0.8, 1, v, 555)
		sphere.addInsideSphere(vector2(0,0), 0, 0.6, -v, 555)

	elif total_layers == 2:
		nnetwork = NNetwork(0.03, 2000)

		layer0 = Layer(nnetwork, "layer0", True, 2)
		layer1 = Layer(nnetwork, "layer1", True, 1)

		layer0.connect(layer1, True)

		sphere.addInsideSphere(vector2(0.5,0.5), 0, 0.5, 1, 22)
		sphere.addInsideSphere(vector2(-0.5,-0.5), 0, 0.5, -1, 22)
		sphere.addInsideSphere(vector2(-0.5,0.5), 0, 0.5, -1, 22)

	elif total_layers == 3:
		nnetwork = NNetwork(0.03, 20000)

		layer0 = Layer(nnetwork, "layer0", True, 2)
		layer1 = Layer(nnetwork, "layer1", True, 4)
		layer2 = Layer(nnetwork, "layer3", True, 1)	
		
		layer0.connect(layer1, True)
		layer1.connect(layer2, True)

		if 0:
			v = 111
			sphere.addInsideSphere(vector2(0.5,0.5), 0, 0.5, v, 222)
			sphere.addInsideSphere(vector2(-0.5,-0.5), 0, 0.5, -v, 222)
			sphere.addInsideSphere(vector2(-0.5,0.5), 0, 0.5, v, 222)
			sphere.addInsideSphere(vector2(0.5,-0.5), 0, 0.5, v, 222)
		else:
			v = 111		#TODO: figure out what this value really means
			centers = [vector2(-0.8, 0.7), vector2(0.8, 0.7), vector2(-0.4, 0.7), vector2(0.4, 0.7)]
			centers2 = [vector2(-0.8, -0.7), vector2(0.8, -0.7), vector2(-0.4, -0.7), vector2(0.4, -0.7)]
			centers3 = [vector2(0, 0.6), vector2(0, 0.3), vector2(0, 0), vector2(0, -0.3), vector2(0, -0.6)]
			centers += centers2 + centers3
			for center in centers:
				sphere.addInsideSphere(center, 0, 0.3, v, 88)
			print("positives = ", len(centers) * 88)
			
			sphere.addInsideSphere(vector2(-0.8, 0), 0, 0.4, -v, 555)
			sphere.addInsideSphere(vector2(0.8, 0), 0, 0.4, -v, 555)

	nnetwork.set_first_layer(layer0)

	nnetwork.reset()

	nnetwork.logInterval = nnetwork.epochs / 5
	nnetwork.train(sphere.Y, sphere.X1, sphere.X2)

	#nnetwork.dump_formula()
	#print("trained w = ", nnetwork.last_layer.nodes[0].iw, "b = ", nnetwork.last_layer.nodes[0].bias)



###############################################################################
##							create_two_feature_layers_linear
###############################################################################
def test():
	#NOTE: currently, must pass (0,0), because b has not been implemented yet
	y0 = random.uniform(-1, 1)
	y1 = random.uniform(-1, 1)
	create_two_feature_layers(vector2(-1,y0), vector2(1,-y1), 200, 0.2)	  #last 2: n, noise

	global mc
	mc = MCanvas(-1, 1, 0.01, -1, 1, 0.01)		# 2/0.01 pixels
	lc = NNetworkForCanvas(nnetwork)
	lc.fill(mc.X, mc.Y, 20)
	mc.Z = lc.Z

	#fela.dump()
	mc.draw_feature(fela.X1, fela.X2, fela.Y)

	mc.plot()

def create_two_feature_layers(p0, p1, n, noise):
	global layer0, layer1, feature, nnetwork, fela

	nnetwork = NNetwork(0.01, 2000)

	#NOTE: currently, must pass (0,0), because b has not been implemented yet
	fela = LineFeatureLabel2(p0, p1, n, noise)	 

	layer0 = Layer(nnetwork, "layer0", 0)
	layer0.newNodes(2)
	layer1 = Layer(nnetwork, "layer1", 0)
	layer1.newNodes(1)

	layer0.nodes[0].name = "i1"
	layer0.nodes[1].name = "i2"
	layer1.nodes[0].name = "o"

	layer0.nodes[0].connect(layer1.nodes[0], 0)
	layer0.nodes[1].connect(layer1.nodes[0], 0)

	layer0.connect(layer1)

	nnetwork.set_first_layer(layer0)
	nnetwork.train(fela.Y, fela.X1, fela.X2)

	print("trained w = ", nnetwork.last_layer.nodes[0].iw, "b = ", nnetwork.last_layer.nodes[0].bias)

###############################################################################
##							create_single_feature_layers
###############################################################################
#NOTE: currently, must pass (0,0), because b has not been implemented yet
def create_single_feature_layers(stopx, stopy):
	global layer0, layer1, feature, nnetwork

	nnetwork = NNetwork(0.01, 2000)

	fela = LineFeatureLabel()			#fela, both feature and label

	stop = vector2(stopx, stopy)
	start = vector2(0,0)
	dir = stop - start
	fela.generate_line_set_2d(start, stop, 200, 0.1)
	#fela.dump()
	print("expect w = ", dir.y/dir.x)

	layer0 = Layer(nnetwork, "layer0", 0)
	layer0.newNodes(1)
	layer1 = Layer(nnetwork, "layer1", 0)
	layer1.newNodes(1)

	layer0.nodes[0].name = "i"
	layer1.nodes[0].name = "o"

	layer0.nodes[0].connect(layer1.nodes[0], 0)

	layer0.connect(layer1)

	nnetwork.set_first_layer(layer0)
	nnetwork.train(fela.Y, fela.X)

	print("trained w = ", nnetwork.last_layer.nodes[0].iw)


