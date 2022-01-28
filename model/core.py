from keras.models import Model, load_model
from keras.layers import Input, LSTM, Permute, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.core import Lambda

def build_model(depth=1):
	# dimension of input should be (batch, latitude, longitude, features)
	inputs = Input( ( 48, 376, 10 ), name='model_input')  # batch dimension omitted here
	s = Lambda(lambda x: x / 1) (inputs) # 48, 376, 10

	c1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block1_Conv1') (s)    # 48, 376
	c1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block1_Conv2') (c1)   # 48, 376
	p1 = MaxPooling2D((2, 2), name='Block1_MaxPool', padding='same') (c1)   # 24, 188


	c2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block2_Conv1') (p1)   # 24, 188
	c2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block2_Conv2') (c2)   # 24, 188
	p2 = MaxPooling2D((2, 2), name='Block2_MaxPool', padding='same') (c2)   # 12, 94


	c3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Block3_Conv1') (p2)   # 12, 94
	c3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Block3_Conv2') (c3)   # 12, 94
	p3 = MaxPooling2D((2, 2), name='Block3_MaxPool', padding='same') (c3)  # 6 x 47

	c4r = Permute((3, 1, 2), name='Block4_Permute1') (p3) # 10 x 6 x 47
	c4r = Reshape((-1, 282), name='Block4_Reshape') (c4r)   # 10 x 282
	f4 = Permute((2, 1), name='Block4_Permute2') (c4r)  # 282 x 10

	lstm = LSTM(1024, return_sequences=True, name='LSTM1') (f4)

	resh = Reshape( (6, 47, 1024) , name='Block5_Reshape') (lstm)

	u5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', name='Block5_UpConv') (resh)  # 12, 94
	u5_comb = concatenate([u5, c3])  # 9 x 12
	c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Block5_Conv1') (u5_comb)  # 12, 94
	c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Block5_Conv2') (c5)  # 12, 94

	u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='Block6_UpConv') (c5)  # 24 x 188
	u6_comb = concatenate([u6, c2])
	c6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block6_Conv1') (u6_comb)  # 24 x 188
	c6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block6_Conv2') (c6)  # 24 x 188

	u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='Block7_UpConv') (c6)  # 48, 376
	u7_comb = concatenate([u7, c1])
	c7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block7_Conv1') (u7_comb)  # 48, 376
	c7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block7_Conv2') (c7)  # 48, 376

	outputs = Conv2D(depth, (1, 1), activation='relu', name='model_output_2') (c7)

	model = Model(inputs=[inputs], outputs=[outputs])

	return model




class RUnet_model():

	def __init__(self, level1, level2):
		self.level1 = level1
		self.level2 = level2
		self.depth = self.level2 - self.level1 + 1
		self.model = build_model(depth=self.depth)

	def compile(self, optimizer, loss, **kwargs):
		self.model.compile(optimizer=optimizer, loss=loss, **kwargs)

	def info(self):
		self.model.summary()

	def train(self, *args, **kwargs):
		self.model.fit( *args, **kwargs )

	def predict(self, x):
		return self.model.predict(x)

	def load_weights(self, filename):
		self.model.load_weights(filename)

	def save_model(self, modelname):
		self.model.save(modelname)