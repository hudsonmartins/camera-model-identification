import random, os.path
import numpy as np

class gradient_descent():

	def __init__ (self, reg):
		self.alpha = 1	 #Learning rate
		self.lamb = 1		#Regularization param
		self.regularization = reg  #True if there is regularization
		#self.mean = mean
		#self.std = std
		
	def fit(self, x_train, y_train):
		print "Training..."

		vec_error = []
		bias = random.randint(-1, 1)	#Initialize the bias as a random value
		theta = []


		for i in range(len(x_train[0])):
			theta.append(random.randint(-1, 1)) #Initialize theta as random values
			
		convergence = False
		prev_cost = 0
		convergence = 0
		n_iterations = 0
		
		while(convergence < 100):	
			n_iterations += 1
			h = []
			m = len(x_train) #number of examples
			for i in range(m):
				h.append(bias)
				for j in range(len(theta)):
					h[i] += theta[j]*x_train[i][j]
				h[i] = self.sigmoid(h[i])

			cost = self.cost(h, y_train) #Calculates the cost	
			if n_iterations % 1 == 0:
				print "Epoch: ", n_iterations ," Cost: ", cost
				
			vec_error.append(cost)
						
			if round(prev_cost, 4) == round(cost, 4):
				convergence += 1
			else:
				convergence = 0				
			gd = self.gradient(h, x_train, y_train) #Calculates the gradient descent
			
			bias = bias - self.alpha*gd[0]
			if self.regularization:
				for i in range(len(gd)-1):
					theta[i] = theta[i] - self.alpha*(gd[i+1] + (self.lamb/m) * theta[i])
			else:
				for i in range(len(gd)-1):
					theta[i] = theta[i] - self.alpha*gd[i+1]
			
			prev_cost = cost

			
		print "Pesos ", bias, theta			
		
		#file_name = raw_input("Insert the file name to save the logistic regression data\n")
		
		file_name = "10classes_correlation"
		theta = np.append([bias], theta, axis=0)
		self.save_results(theta, vec_error, file_name, n_iterations)
				
		return theta
		
	def cost(self, h, y):	
		j = 0 #The cost J
		m = len(h) #The number of examples

		#print h
		for i in range(m):
			if(y[i] == 1):
				if(h[i] == 0):
					j += -1.0*np.log(0.00000001)			
				else:
					j += -1.0*np.log(h[i])			
			else:
				if(h[i] == 1):
					j += -1.0*np.log(0.00000001)			
				else:
					j += -1.0*np.log(1-h[i])					
		j *= (1.0/m)
		return j 	
		
	def gradient(self, h, x, y):
		gd = []
		m = len(h) #The number of examples

		#1/m * SUM of (hi -yi)
		bias_err = 0
		for i in range(m):
			bias_err += h[i]-y[i]

		gd.append(bias_err/m)
		
		for i in range(len(x[0])):
			err = 0
			for j in range(m):
				err += (h[j]-y[j])*x[j][i]
			gd.append(err/m)	
			
		return gd

	def sigmoid(self, z):
		#Prevent overflow
		z = np.clip(z, -500, 500 )
		return 1.0/(1.0 + np.exp(-1.0*z))
	
	def save_results(self, weights, error, file_name, n_iterations):
		if not os.path.isdir('results'):
			os.mkdir('results')
			
		if os.path.isfile('results/'+file_name):
			myfile = open('results/'+file_name, 'a+')
			myfile.write('\n-----------------------------------------------\n')
			myfile.write('\nLearning Rate: '+ str(self.alpha))
			#myfile.write('\nMean: '+ str(self.mean))
			#myfile.write('\nStd: '+ str(self.std))
			myfile.write('\nWeights: '+str(weights))
			myfile.write('\nNumber of iterations: '+str(n_iterations))
			myfile.write('\nFinal cost: '+str(error[len(error)-1]))
			

			myfile.close()
		else:
			myfile = open('results/'+file_name, 'w+')
			myfile.write('\n-----------------------------------------------\n')
			myfile.write('\nLearning Rate: '+ str(self.alpha))
			#myfile.write('\nMean: '+ str(self.mean))
			#myfile.write('\nStd: '+ str(self.std))
			myfile.write('\nWeights: '+str(weights))
			myfile.write('\nNumber of iterations: '+str(n_iterations))
			myfile.write('\nFinal cost: '+str(error[len(error)-1]))
			myfile.close()
			
