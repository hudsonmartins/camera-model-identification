from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from sklearn.feature_extraction import image as sklimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import feature_extraction, gradient_descent, mlp
import glob, pywt, csv, os.path, os, cv2, keras.models, keras.utils

	   
def get_fingerprint(classes):
	pattern_list = []
	for i in range(len(classes)):
		print "Getting fingerprint for "+classes[i]
		pattern_list.append(np.load('fingerprint_'+classes[i]+'.npy'))

	return pattern_list
	
def calculate_features(I_noise):	
	feat = []
	
	for pattern in fingerprint:
		#Get the correlations R-R, G-G, B-B 
		corr_r, corr_g, corr_b = feature_extraction.get_correlation(pattern, I_noise)
		feat.append(corr_r)
		feat.append(corr_g)
		feat.append(corr_b)
		#Get the cross-correlations R-G, R-B, G-B, G-R, B-G, B-R,  	
	#	cross_corr = feature_extraction.get_cross_correlation(pattern, I_noise)
	#	for corr in cross_corr:
	#		feat.append(corr)
	"""
	statistical = feature_extraction.get_statistical_features(I_noise)
	
	for stat in statistical:
		feat.append(stat)
	"""
	return feat

def get_train_features(fingerprint, classes):
	print "Getting features"
	train_file = np.genfromtxt('features/train_features.csv',delimiter=',')
	targ = []
	train_feat = []
	n_rows = 0
	for row in train_file:
		n_rows += 1
		feat = []		
		for i in range(len(row)-1):
			if (i < 3 or (i > 8 and i < 12) or (i > 17 and i < 21)	or (i > 26 and i < 30)	or (i > 35 and i < 39)	or (i > 44 and i < 48) or (i > 53 and i < 57) or (i > 62 and i < 66) or (i > 71 and i < 75) or (i > 80 and i < 84)):	
				feat.append(row[i])
				
		train_feat.append(feat)	
		targ.append(int(row[len(row)-1]))
		
	return train_feat, targ


def logistic(train_feat, train_targ):

	print "Training..."		
	logisticRegr = LogisticRegression()
	h = logisticRegr.fit(train_feat, train_targ)
	return h


def onevsall(feat, classes):
	n_classes = len(classes)
	h = []
	
	for i in range(n_classes):
		train_feat = []
		train_targ = []
		
		for camera in feat:
			for image in feat[camera]:
				train_feat.append(image)
				if camera == classes[i]:
					train_targ.append(1)
				else:
					train_targ.append(0)
					
		#feat_std, feat_mean = get_stdnmean(train_feat, len(train_feat[0]))
		#train_feat = normalize(train_feat, feat_mean, feat_std, len(train_feat[0]), len(train_feat))
		gd = gradient_descent.gradient_descent(reg=False)
		h.append(gd.fit(train_feat, train_targ))
	return h

def sigmoid(z):
		#Prevent overflow
		z = np.clip(z, -500, 500 )
		return 1.0/(1.0 + np.exp(-1.0*z))

def neural_net(train_feat, train_targ, classes):
	n_classes = len(classes)
	NN = mlp.mlp()
	NN.train(np.array(train_feat), np.array(train_targ), classes)

		
def validate(models, fingerprint, classes):
	print "Validating model"
	count_cam = 0
	extensions = ('/*.jpg', '/*.JPG')
	for camera in classes:
		features = []
		n_acertos = 0
		for ex in extensions:
			for image in glob.glob('dataset/train/'+camera+'/validation/'+ex):
				#Treating image 
				img = Image.open(image)
				if img.size[0] > img.size[1]:
					img = img.rotate(90)
				center = [img.size[0]/2, img.size[1]/2]
				img = img.crop((center[0]-256, center[1]-256, center[0]+256, center[1]+256))
				img = np.asarray(img)
		
				#Getting noise image
				I_noise = feature_extraction.get_noise(img)
				I_noise = feature_extraction.increase_green_channel(I_noise)
				#Extracting features
				feat = calculate_features(I_noise)

				pred=[]
				row = 0
				for h in models:
					pred.append(h[0])
					for i in range(len(feat)):
						pred[row] += h[i+1] * feat[i]
					pred[row] = sigmoid(pred[row])
					row += 1
				print "--------------------------------------------------"
				print "Predictions for ", camera,": ", pred
				if (np.argmax(pred) == count_cam):
					n_acertos+=1
					print "Acerto ", n_acertos

				
		count_cam+=1

		#print models.predict(features)
			
def predict(models, classes, fingerprint):
	created = False

	results = []
	
	for image in glob.glob('dataset/test/*.tif'): 
	
		img = Image.open(image)
		img = np.asarray(img)
		
		#Getting noise image
		I_noise = feature_extraction.get_noise(img)
		I_noise = feature_extraction.increase_green_channel(I_noise)
		
		feat = calculate_features(I_noise)
		
		#print "Features test data: ", feat
		pred = []
		
		row = 0
		for h in models:
			pred.append(h[0])
			for i in range(len(feat)):
				pred[row] += h[i+1] * feat[i]
			pred[row] = sigmoid(pred[row])
			row += 1
		
		print "Predictions: ", pred		
		camera = np.argmax(pred)
		#results.append(camera)

		count = 1
		while (not created):
			fn = 'results/predictions'+str(count)+'.csv'
			if os.path.isfile(fn): 
				count += 1
			else:
				created = True
				
		print "Writing on csv file"
		with open(fn, 'a') as csvfile:
			writer = csv.writer(csvfile, delimiter = ',', quoting=csv.QUOTE_NONE)
			writer.writerow([image, classes[camera]])
	
def validate_nn(model):
	print "Validating model"
	valid_feat = []
	valid_targ = []
	for fname in glob.glob('features/validation_features*.csv'):
		validation_file = np.genfromtxt(fname,delimiter=',')
		for row in validation_file:
			targ = []
			feat = []
			targ.append(int(row[(len(row)-1)]))
			for i in range(len(row)-1):
				feat.append(row[i])
			
			valid_feat.append(feat)
			valid_targ.append(targ)
			
	valid_feat = np.array(valid_feat)
	valid_targ = np.array(valid_targ)
	valid_targ = keras.utils.to_categorical(valid_targ)
	score = model.evaluate(valid_feat, valid_targ)		
	print('Validation loss:', score[0])
	print('Validation accuracy:', score[1])

def validate_logistic(model):
	print "Validating model"
	valid_feat = []
	valid_targ = []
	for fname in glob.glob('features/validation_features*.csv'):
		validation_file = np.genfromtxt(fname,delimiter=',')
		for row in validation_file:
			targ = []
			feat = []
			targ.append(int(row[(len(row)-1)]))
			for i in range(len(row)-1):
				if (i < 3 or (i > 8 and i < 12) or (i > 17 and i < 21)	or (i > 26 and i < 30)	or (i > 35 and i < 39)	or (i > 44 and i < 48) or (i > 53 and i < 57) or (i > 62 and i < 66) or (i > 71 and i < 75) or (i > 80 and i < 84)):
					feat.append(row[i])
			
			valid_feat.append(feat)
			valid_targ.append(targ)
	
	valid_feat = np.array(valid_feat)
	valid_targ = np.array(valid_targ)
	y_pred = model.predict(valid_feat)
	#print("Predicted class %s, real class %s" % (y_pred, valid_targ))
	#print ("Probabilities for each class from 1 to 10: %s"% model.predict_proba(valid_feat))
	print('Accuracy of logistic regression classifier on validation set: {:.2f}'.format(model.score(valid_feat, valid_targ)))
	print confusion_matrix(valid_targ,y_pred)
	print classification_report(valid_targ, y_pred)

def predict_logistic(model, classes):
	print "Predicting for test set"
	test_feat = []
	label = []
	created = False
	
	for fname in glob.glob('features/test_features*.csv'):
		test_file = np.genfromtxt(fname,delimiter=',', dtype = None)
		for row in test_file:
			feat = []
			label.append(row[0])

			for i in range(1, len(row)):
				if (i < 4 or (i > 9 and i < 13) or (i > 18 and i < 22)	or (i > 27 and i < 31)	or (i > 36 and i < 40)	or (i > 45 and i < 49) or (i > 54 and i < 58) or (i > 63 and i < 67) or (i > 72 and i < 76) or (i > 81 and i < 85)):	
					feat.append(row[i])
			
			test_feat.append(feat)
	
	y_pred = model.predict(test_feat)
	
	count = 1
	while (not created):
		fn = 'results/predictions'+str(count)+'.csv'
		if os.path.isfile(fn): 
			count += 1
		else:
			created = True
			
	print "Writing on csv file"
	for i in range(len(y_pred)):
		with open(fn, 'a') as csvfile:
			writer = csv.writer(csvfile, delimiter = ',', quoting=csv.QUOTE_NONE)
			writer.writerow([label[i], classes[y_pred[i]]])

def predict_nn(model, classes):
	print "Predicting for test set"
	test_feat = []
	label = []
	camera = []
	created = False
	
	for fname in glob.glob('features/test_features*.csv'):
		test_file = np.genfromtxt(fname,delimiter=',', dtype = None)
		for row in test_file:
			feat = []
			label.append(row[0])

			for i in range(1, len(row)):
				feat.append(row[i])
			
			test_feat.append(feat)
	test_feat = np.array(test_feat)
	y_pred = model.predict(test_feat)
	
	for pred in y_pred:
		camera.append(np.argmax(pred))

	count = 1
	while (not created):
		fn = 'results/predictions'+str(count)+'.csv'
		if os.path.isfile(fn): 
			count += 1
		else:
			created = True
			
	print "Writing on csv file"
	for i in range(len(camera)):
		with open(fn, 'a') as csvfile:
			writer = csv.writer(csvfile, delimiter = ',', quoting=csv.QUOTE_NONE)
			writer.writerow([label[i], classes[camera[i]]])


			
#--------------------All the classes in the training data----------------------------
classes = ['HTC-1-M7',
	   'iPhone-4s',
   	   'iPhone-6',
   	   'LG-Nexus-5x',
   	   'Motorola-Droid-Maxx',
   	   'Motorola-Nexus-6',
   	   'Motorola-X',
   	   'Samsung-Galaxy-Note3',
   	   'Samsung-Galaxy-S4',
   	   'Sony-NEX-7']

fingerprint = get_fingerprint(classes)	   
feat, targ = get_train_features(fingerprint, classes)

answer = raw_input("Want to train with logistic regression? (Y or N)\n")
if(answer == 'y' or answer == 'Y'):
	h = logistic(feat, targ)
	validate_logistic(h)
	answer = raw_input("Want to predict for test data? (Y or N)\n")
	if(answer == 'y' or answer == 'Y'):
		predict_logistic(h, classes)
			
answer = raw_input("Want to train with neural networks? (Y or N)\n")
if(answer == 'y' or answer == 'Y'):
	neural_net(feat, targ, classes)
	model = keras.models.load_model('results/network2')
	validate_nn(model)
	answer = raw_input("Want to predict for test data? (Y or N)\n")
	if(answer == 'y' or answer == 'Y'):
		predict_nn(model, classes)			





"""
fig, ax = plt.subplots()
ax.imshow(fingerprint)
plt.show()
"""
