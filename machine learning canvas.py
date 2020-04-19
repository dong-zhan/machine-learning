# color of the canvas
# 1, get predicated value for all examples
# 2, compare the predicated with example value, put the difference in each pixel
# 3, find all known differences for current pixel in a neighborhood
# 4, weighted average as current pixel value, use distance as weight?

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

###############################################################################
##                          LineForCanvas
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
        #if v-y > 0:            #this splits the space into 2 halves with different colors.
            return self.fillColor
        else:
            return 0

    def fill(self, X, Y, color) :
        self.fillColor = color
        self.Z = self.vFillFunc(X, Y)


def test():
    global mc
    #mc = Canvas(-6, 6, 0.01, -1, 1, 0.01)          #test sigmoid
    mc = Canvas(-1, 1, 0.01, -1, 1, 0.01)   
    lc = LineForCanvas(1, 0, 0.01)
    lc.fill(mc.X, mc.Y, 20)
    mc.Z = lc.Z
    mc.plot()


###############################################################################
##                          MCanvas
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
##                          NNetworkForCanvas
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
        if v > 0 :
            return 0.5
        return -0.5

    def fill(self, X, Y, color) :
        self.fillColor = color
        self.Z = self.vLayerFunc(X, Y)


def test():
    create_two_feature_layers(vector2(-1,-1), vector2(1,1), 200, 0.2)

    global mc
    mc = MCanvas(-1, 1, 0.01, -1, 1, 0.01)
    lc = NNetworkForCanvas(nnetwork)
    lc.fill(mc.X, mc.Y, 20)
    mc.Z = lc.Z

    #fela.dump()
    mc.draw_feature(fela.X1, fela.X2, fela.Y)

    mc.plot()




###############################################################################
##                          create_two_feature_layers
###############################################################################
def create_two_feature_layers(p0, p1, n, noise):
    def sigmoid(x):
	    return x
	
    def d_sigmoid(x):
	    return 1

    global layer0, layer1, feature, nnetwork, fela

    nnetwork = NNetwork(0.01, 2000)

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

    print("trained w = ", nnetwork.last_layer.nodes[0].iw)


create_two_feature_layers(vector2(0, 0), vector2(1,1), vector2(0,0), vector2(-1,0), 20, 0)

###############################################################################
##                          create_single_feature_layers
###############################################################################
def create_single_feature_layers(stopx, stopy):
    def sigmoid(x):
	    return x
	
    def d_sigmoid(x):
	    return 1

    global layer0, layer1, feature, nnetwork

    nnetwork = NNetwork(0.01, 2000)

    fela = LineFeatureLabel()           #fela, both feature and label

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


