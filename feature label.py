import random
import math

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
		pos += cross * side * noise		
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
			self.Side[i] = side * 100		#TODO: *100 gives much more accurate result, why? -> I guess, *100 gives a bigger 
											#difference between 2 labels.  (for example, 0.3, -0.3 are too close)

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
#SphereFeatureLabel holds one feature and one label which are close to a line
###############################################################################
class SphereFeatureLabel:
	def __init__(self):
		self.X1 = []			#feature
		self.X2 = []			#label
		self.Y = []

	def addInsideSphere(self, center, noise0, noise1, y, cnt) :		#other label can be added in a second pass
		for i in range(cnt):
			dir = vector2()
			dir.getRandomUnitVector()
			r = random.uniform(noise0, noise1)
			dir *= r
			dir = dir + center

			self.Y.append(y)
			self.X1.append(dir.x)
			self.X2.append(dir.y)
		
	def dump(self):
		ly = len(self.Y)
		for i in range(ly):
			print(self.X1[i], self.X2[i], self.Y[i])


def test():
	global sphere, mc
	sphere = SphereFeatureLabel()
	sphere.addInsideSphere(vector2(0,0), 0.8, 1, 100, 220)
	sphere.addInsideSphere(vector2(0,0), 0, 0.6, -100, 220)

	mc = MCanvas(-1, 1, 0.01, -1, 1, 0.01)      # 2/0.01 pixels
	mc.fill(0)
	mc.draw_feature(sphere.X1, sphere.X2, sphere.Y)

	mc.plot()
