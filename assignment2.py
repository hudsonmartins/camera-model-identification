from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
import glob, pywt, feature_extraction, gradient_descent, csv

	   
def get_features(classes):
	img_list = []
	noise_list = []
	denoised_list = []
	feat_dict = dict()
	for i in range(len(classes)):
		count_img = 0   
		print "Getting images and calculating noise for "+classes[i]
		extensions = ('/*.jpg', '/*.JPG')
		for ex in extensions:
			for image in glob.glob('dataset/train/'+classes[i]+ex): 
				count_img += 1
				#if count_img > 1:
				#	break
		
				print "image: ", count_img
				img = Image.open(image).convert('RGB')

				if img.size[0] > img.size[1]:
					img = img.rotate(90, expand=True)
	
				img = np.asarray(img)/255.0
				I_noise, denoised = feature_extraction.get_noise(img)
	
				#img_list.append(img)
				#noise_list.append(I_noise)
	
				feat = feature_extraction.extract_features_moments(I_noise)
				if(classes[i] in feat_dict):
					feat_dict[classes[i]].append(feat)
				else:
					feat_dict[classes[i]] = [feat]
	return feat_dict

def z_norm(x, mean, std):
	"""
		Calculates the z-norm
	"""			
	x = (x - mean)/std

	return x
	
def normalize(dataset, mean, std, n_feat, n_examples):	
	"""
		Returns the z-norm of the dataset
	"""

	if n_examples > 1:	
		for i in range(n_examples):
			for j in range(n_feat):
				dataset[i][j] = z_norm(dataset[i][j], mean[j], std[j])
	else:
		for i in range(n_feat):
			dataset[i] = z_norm(dataset[i], mean[i], std[i])
		
	return dataset

def get_stdnmean(dataset, n_feat):
	"""
		Calculates the standard deviation and mean for a given dataset
	"""
	mean = []
	std = []
	for col in range(n_feat):
		#print col
		feat = []
		for example in dataset:
			feat.append(example[col])
		mean.append(np.mean(feat))
		std.append(np.std(feat))

		
	return std, mean
	
def onevsall(feat, classes):
	n_classes = len(classes)
	h = []
	
	for i in range(n_classes):
		train_feat = []
		train_targ = []
		
		for camera in feat:
			for image in feat[camera]:
				#Each image has 63 features	
				train_feat.append(image)
				if camera == classes[i]:
					train_targ.append(1)
				else:
					train_targ.append(0)
					
		feat_std, feat_mean = get_stdnmean(train_feat, len(train_feat[0]))
		train_feat = normalize(train_feat, feat_mean, feat_std, len(train_feat[0]), len(train_feat))
		gd = gradient_descent.gradient_descent(reg=False, mean=feat_mean, std=feat_std)
		h.append(gd.fit(train_feat, train_targ))
	return h, feat_std, feat_mean

def sigmoid(z):
		return 1.0/(1.0 + np.exp(-1.0*z))
		
def predict(models, classes, feat_std, feat_mean):
	feat_dict = dict()

	for image in glob.glob('dataset/test/*.tif'): 
	
		img = Image.open(image).convert('RGB')

		if img.size[0] > img.size[1]:
			img = img.rotate(90, expand=True)

		img = np.asarray(img)/255.0
		I_noise, denoised = feature_extraction.get_noise(img)
		
		feat = feature_extraction.extract_features_moments(I_noise)
		feat = normalize(feat, feat_mean, feat_std, len(feat), 1)
		
		print "Features test data: ", feat
		pred = []
		
		row = 0
		for h in models:
			pred.append(h[0])
			for i in range(len(feat)):
				pred[row] += h[i+1] * feat[i]
			pred[row] = sigmoid(pred[row])
			row += 1
			
		camera = np.argmax(pred)
		fn = 'results/predictions.csv'
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

	   
feat = get_features(classes)
h, std, mean = onevsall(feat, classes)
predict(h, classes, std, mean)

"""
print "Calculating Fingerprint for the model "+classes[0]
fingerprint = feature_extraction.get_fingerprint(noise_list, img_list)
feature_extraction.extract_features(fingerprint)

fig, ax = plt.subplots()
ax.imshow(fingerprint)
plt.show()
"""
