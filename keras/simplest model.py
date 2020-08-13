def sum_layer(layer):
	print("len weights(this incudes b):", len(layer.weights))
	print("bp adjust weights:", layer.weights[0].numpy()[0], "b", layer.weights[1].numpy()[0])
	#print("len trainable_weights:", len(layer.trainable_weights))
	print("fp adjust type(output):", type(layer.output))
	
	#output = keras_function([training_data, 1])
	#print(output)
	
def get_value(v):
	i = np.array([v])
	print(func(i))
	print(model(i))
	print(model.predict(i))	
	

def create_nw():
	from keras.optimizers import SGD

	global model, functors, func, layer
	model = tf.keras.models.Sequential()
	model.add(tf.keras.Input(shape=(1,)))							#input layer
	model.add(tf.keras.layers.Dense(1, activation='relu'))   		#output layer
	
	#opt = SGD(lr=0.01)
	model.compile(loss='MeanSquaredError', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])  #MeanSquaredError binary_crossentropy
	
	model.summary()
	print("learning rate:", K.eval(model.optimizer.lr))
	
	layer = model.layers[0]

	func = K.function([model.input], [model.layers[0].output])		#func accepts only np array
	
def create_middle_model(model, layer):
	intermediate_layer_model = keras.Model(inputs=model.input,
										   outputs=layer.output)
	return intermediate_layer_model(data)
	
def info():
	global layer
	layer = model.layers[0]
	sum_layer(layer)

def classify_point_2d(x, b):	#recreate model if this function has been changed.
	if(x>b):
		return 0.1
	return -0.1
	
def generate_train_data(b, n):		#train data x, lable y
	X = []
	Y = []
	for i in range(n):
		x = random.uniform(-1, 1)
		y = classify_point_2d(x, b)
		X.append(x)
		Y.append(y)
	return X, Y
	
def getY(x, w, b):
	return (x*w)+b
	
def train(b, n, verbose = 0):
	x, y = generate_train_data(b, n)
	if verbose :
		print(x, y)
	history = model.fit(x, y, epochs=1, verbose=0)	#x input data, y target data
	sum_layer(layer)
	get_value(0)
	
	w, b = (layer.weights[0].numpy()[0], layer.weights[1].numpy()[0])

	plt.scatter(x, y, marker='o', color="red")
	plt.plot([-2, 2], [getY(-2, w, b), getY(2, w, b)])
	plt.show()
