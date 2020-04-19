import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Canvas:
    def __init__(self, x0, x1, xstep, y0, y1, ystep):
        global canvas_vFillFunc
        self.x0 = x0
        self.x1 = x1
        self.w = x1 - x0            #width in real world space
        self.pw = self.w / xstep    #width in pixel space
        self.w2pw = self.w * self.pw
        self.xstep = xstep
        self.y0 = y0
        self.y1 = y1
        self.h = y1 - y0
        self.ph = self.h / ystep
        self.ystep = ystep
        self.x = np.arange(x0, x1, xstep)           
        self.y = np.arange(y0, y1, ystep)           
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.fillColor = 0
        self.vFillFunc = np.vectorize(self.fillFunc, excluded='self')
        self.Z = None

    def ndcToScreen(self, x, y):
        return (x-self.x0)/self.w * self.pw, (y-self.y0)/self.h * self.ph

    def setColor(self, x, y, color):          #use real world coordinates
        nsX, nsY = self.ndcToScreen(x, y)
        nsX += 0.5
        nsY += 0.5
        #print(int(nsX), int(nsY))
        nsY = int(nsY)
        nsX = int(nsX)
        if nsY >= self.pw or nsX >= self.ph or nsY < 0 or nsX < 0 :
            return

        self.Z[nsY][nsX] = color

    def setColorPC(self, x, y, color):        #use pixel coordinates
        self.Z[y][x] = color

    #this can be done with vectorized function too, but, that's like a fullscreen pass, this one limits to pixels on line
    #so, this is should be faster (compare to LineForCanvas(vectorized version))
    def drawLine(self, a, b, color):        #a, b are both vector2
        dir = b - a
        cnt = 0
        stepx = 0
        stepy = 0

        if abs(dir.y) > abs(dir.x) : #along y
            slope = dir.x / dir.y
            stepy = self.ystep
            cnt = dir.y / self.ystep
            if(a.y > b.y):
                stepy *= -1
                cnt -= 0.5
            else:
                cnt += 0.5
            stepx = stepy * slope

        else : #along x
            slope = dir.y / dir.x
            stepx = self.xstep
            cnt = dir.x / self.xstep
            if(a.x > b.x):
                stepx *= -1
                cnt -= 0.5
            else:
                cnt += 0.5
            stepy = stepx * slope

        cnt = abs(int(cnt))
        for i in range(cnt):
            x = a.x + stepx * i
            y = a.y + stepy * i
            self.setColor(x, y, color)

    def fillFunc(self, a, b):
        return self.fillColor

    def fill(self, color) :
        self.fillColor = color
        self.Z = self.vFillFunc(self.X, self.Y)

    def shape(self):
        return self.X.shape

    def plot(self):
        fig, ax = plt.subplots()
        #The orientation of the image in the final rendering is controlled by the origin and extent
        im = ax.imshow(self.Z, interpolation='bilinear', cmap=cm.RdYlGn,
                       origin='lower', extent=[-1, 1, -1, 1],
                       vmax=abs(self.X).max(), vmin=-abs(self.X).max())
        plt.show()



            
def test():
    global c
    c = Canvas(-1, 1, 0.01, -1, 1, 0.01)
    c.fill(0)

    #Canvas.drawLine = drawLine
    c.drawLine(vector2(-22,0), vector2(22,-1), 20)

#    for x in range(20) :
#        c.setColorPC(x, 0, 20)

#    for x in np.arange(-1, 1, 0.1):
#        c.setColor(x, -2, 20)

    c.plot()



###################### vectorize #############################
def myfunc(a, b):
    "Return a-b if a>b, otherwise return a+b"
    if a > b:
        return a - b
    else:
        return a + b

def vectorize():
    vfunc = np.vectorize(myfunc)
    arr = vfunc([1, 2, 3, 4], 2)
    print(arr)
