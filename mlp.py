from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import to_categorical
class mlp():
	def __init__ (self):
		self.size_hidden_layer = 200
				
	def createModel(self, input_dim, n_classes):
		model = Sequential()
		model.add(Dense(self.size_hidden_layer, input_dim=input_dim))
		model.add(Activation('relu'))
		model.add(Dense(self.size_hidden_layer, input_dim=input_dim))
		model.add(Activation('relu'))
		#model.add(Dropout(0.15))
		model.add(Dense(n_classes))
		model.add(Activation('softmax'))
	
		return model


	def train(self, X_train, y_train, classes):
		
		input_dim = len(X_train[0])
		n_classes = len(classes)
		
		#convert to cathegorical
		y_train = to_categorical(y_train) #Convert to categorical variables
		
		self.model = self.createModel(input_dim, n_classes)
		self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

		print("Training...")
		self.model.fit(X_train, y_train, epochs=600, batch_size=32, verbose=2)
		self.model.save('results/network1')
		
	def predict(self, X_test, y_test):
		y_test = to_categorical(y_test)
		score = self.model.evaluate(X_test, y_test) #Evaluate results for test data
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
