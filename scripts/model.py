from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate

def define_model():
	inputs = Input((1, 10), name='inputs')
	d1 = Dense(128, activation="linear")(inputs) 
	c1 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(d1)
	c2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(c1)

	d2 = Dense(256, activation="linear")(c2) 
	c3 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(d2)
	c4 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(c3)

	d3 = Dense(512, activation="linear")(c4) 
	c5 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(d3)
	c6 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(c5)


	lstm = LSTM(512, return_sequences=True, name='LSTM1') (c6)

	d4 = Dense(512, activation="linear")( lstm ) 
	u0 = concatenate([c6, d4])
	c7 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(u0)
	c8 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(c7)

	d5 = Dense(512, activation="linear")(c8) 
	u1 =  concatenate([c4, d5])
	c9 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(u1)
	c10 = Conv1D(512, 2, strides=1, padding='same', activation="relu")(c9)

	d6 = Dense(256, activation="linear")(c10) 
	u2 = concatenate([c2, d6])
	c11 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(u2)
	c12 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(c11)

	d7 = Dense(256, activation="linear")(c12) 
	outputs = Dense(48, activation="linear")(d7) 

	model = Model(inputs=inputs, outputs=outputs)

	return model

class SOC_model():

	def __init__(self):
		self.model = define_model()

	def compile(self, optimizer, loss, **kwargs):
		self.model.compile(optimizer=optimizer, loss=loss, **kwargs)

	def info(self):
		self.model.summary()

	def train(self, *args, **kwargs):
		self.model.fit( *args, **kwargs )

	def predict(self, x):
		return self.model.predict(x)

	def summary(self):
		self.model.summary()

	def load_weights(self, filename):
		self.model.load_weights(filename)

	def save_model(self, modelname):
		self.model.save(modelname)
