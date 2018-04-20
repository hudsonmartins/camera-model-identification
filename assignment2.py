from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from sklearn.feature_extraction import image as sklimage
from sklearn.linear_model import LogisticRegression
import feature_extraction, gradient_descent, mlp
import glob, pywt, csv, os.path, os, cv2

	   
def get_fingerprint(classes):
	pattern_list = []

	for i in range(len(classes)):
		noise_list = []
		count_img = 0   
		print "Getting fingerprint for "+classes[i]
		extensions = ('/*.jpg', '/*.JPG')
		for ex in extensions:
			for image in glob.glob('dataset/train/'+classes[i]+ex): 
				count_img += 1
				if count_img > 50:
					break
		
				#print "image: ", count_img
				img = Image.open(image)

				if img.size[0] > img.size[1]:
					img = img.rotate(90)
				
				center = [img.size[0]/2, img.size[1]/2]
				
				img = img.crop((center[0]-256, center[1]-256, center[0]+256, center[1]+256))
				img = np.asarray(img)
						
				#patches = sklimage.extract_patches_2d(img, (256, 256), max_patches=8)
				#print "Patches, ", patches.shape
				#for patch in patches:
				I_noise = feature_extraction.get_noise(img)
				I_noise = feature_extraction.increase_green_channel(I_noise)
				noise_list.append(I_noise)	
					
		pattern_list.append(feature_extraction.get_pattern(noise_list))

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
		cross_corr = feature_extraction.get_cross_correlation(pattern, I_noise)
		for corr in cross_corr:
			feat.append(corr)
	"""
	statistical = feature_extraction.get_statistical_features(I_noise)
	
	for stat in statistical:
		feat.append(stat)
	"""
	return feat

def get_train_features(fingerprint, classes):
	train_file = np.genfromtxt('features1.csv',delimiter=',')
	feat_dict = dict()
	feat = []

	n_rows = 0
	for row in train_file:
		n_rows += 1
		
		for i in range(len(row-1)):				
			feat.append(i)

		if(classes[(row[len(row)-1])] in feat_dict):
			feat_dict[classes[(row[len(row)-1])]].append(feat)
		else:
			feat_dict[classes[(row[len(row)-1])]] = [feat]
			
	return feat_dict


def logistic(feat, classes):
	print "Training..."
	n_classes = len(classes)
	train_feat = []
	train_targ = []		
	x_train=[]
	for i in range(n_classes):			
		for camera in feat:
			for image in feat[camera]:
				train_feat.append(image)
				train_targ.append(i)
		
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
				#Each image has 30 features	
				train_feat.append(image)
				if camera == classes[i]:
					train_targ.append(1)
				else:
					train_targ.append(0)
					
		#feat_std, feat_mean = get_stdnmean(train_feat, len(train_feat[0]))
		#train_feat = normalize(train_feat, feat_mean, feat_std, len(train_feat[0]), len(train_feat))
		gd = gradient_descent.gradient_descent(reg=True)
		h.append(gd.fit(train_feat, train_targ))
	return h

def sigmoid(z):
		#Prevent overflow
		z = np.clip(z, -500, 500 )
		return 1.0/(1.0 + np.exp(-1.0*z))

def neural_net(feat, classes):
	n_classes = len(classes)
	train_feat = []
	train_targ = []		

	for i in range(n_classes):			
		for camera in feat:
			for image_feat in feat[camera]:
				train_feat.append(image_feat)
				train_targ.append(i)
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
feat = get_train_features(fingerprint, classes)
#h = onevsall(feat, classes)
#h = logistic(feat, classes)
#validate(h, fingerprint, classes)
neural_net(feat, classes)


answer = raw_input("Want to predict for test data? (Y or N)\n")

if(answer == 'y' or answer == 'Y'):
	predict(h, classes, fingerprint)

"""
fig, ax = plt.subplots()
ax.imshow(fingerprint)
plt.show()
"""
