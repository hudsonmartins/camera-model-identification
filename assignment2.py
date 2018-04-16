from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from sklearn.feature_extraction import image as sklimage
from sklearn.linear_model import LogisticRegression
import glob, pywt, feature_extraction, gradient_descent, csv, os.path

	   
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
	
def get_features(fingerprint, classes):
	print "Finding features..."
	feat_dict = dict()
	for i in range(len(classes)): 
		count_img = 0
		print "Getting images and calculating noise for "+classes[i]
		extensions = ('/*.jpg', '/*.JPG')
		for ex in extensions:
			for image in glob.glob('dataset/train/'+classes[i]+ex): 
				count_img += 1
				#if count_img > 50:
				#	break
	
				print "image: ", count_img
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
				feat = []
				for pattern in fingerprint:
				#print "Getting features for image ", count 
					feat_r, feat_g, feat_b = feature_extraction.get_correlation(pattern, I_noise)
					feat.append(feat_r)
					feat.append(feat_g)
					feat.append(feat_b)		
				
				if(classes[i] in feat_dict):
					feat_dict[classes[i]].append(feat)
				else:
					feat_dict[classes[i]] = [feat]
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
	
	for i in range(4, n_classes):
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
		gd = gradient_descent.gradient_descent(reg=False)
		h.append(gd.fit(train_feat, train_targ))
	return h

def sigmoid(z):
		#Prevent overflow
		z = np.clip(z, -500, 500 )
		return 1.0/(1.0 + np.exp(-1.0*z))
		
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
				feat = []
				for pattern in fingerprint:
					feat_r, feat_g, feat_b = feature_extraction.get_correlation(pattern, I_noise)
					feat.append(feat_r)
					feat.append(feat_g)
					feat.append(feat_b)
					
				#features.append(feat)
				

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
	for image in glob.glob('dataset/test/*.tif'): 
	
		img = Image.open(image)
		img = np.asarray(img)
		
		#Getting noise image
		I_noise = feature_extraction.get_noise(img)
		I_noise = feature_extraction.increase_green_channel(I_noise)
		
		feat = []	
		for pattern in fingerprint:
			#print "Getting features for image ", count 
			feat_r, feat_g, feat_b = feature_extraction.get_correlation(pattern, I_noise)
			feat.append(feat_r)
			feat.append(feat_g)
			feat.append(feat_b)	
		
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


		count = 1
		while (not created):
			fn = 'results/predictions'+str(count)+'.csv'
			if os.path.isfile(fn): 
				count += 1
			else:
				created = True
		
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
#feat = get_features(fingerprint, classes)
#h = onevsall(feat, classes)
#h = logistic(feat, classes)

h = [[-2.08221361,  42.20217032,  41.81181685,  41.57915352, -12.13931716,
 -12.08667554, -12.57484775, -12.92297863, -13.33815483, -12.9947462,
  -7.30659693,  -7.52419428,  -8.39603208, -11.38186978, -10.35401577,
 -11.2241588,   -7.38639907,  -9.24936783,  -8.67030193,  -8.27024189,
  -9.00052159,  -5.99880213,  -9.29504344, -10.73434327, -10.12638669,
  -8.45778275,  -8.66483309, -12.00139003, -11.8380798,   -7.35094091,
 -12.27340628], [ -3.63345421,  -4.37482854,  -3.66918993,  -5.10889881,  58.09681922,
  56.41741171,  57.59591631,  -5.4513468,   -5.80717292, -5.63714482,
  -2.92637322,  -2.57423219,  -2.87812188, -12.2906434,   -9.30474196,
  -9.84779479,  -3.58355929,  -2.19712998,  -2.18086723,  -5.53035995,
  -5.73331498,  -4.52926507,  -4.61946599,  -3.95740568,  -3.83965889,
  -8.33075985,  -6.7396395,  -7.90789015, -10.44128288, -5.61068796,
  -8.83128833], [-3.36093178,  -5.36664093,  -4.23597518,  -6.30847524,  -6.32305329,
  -7.67171227,  -6.70067322,  53.02032175,  52.1810667,   53.24076152,
  -3.59884712,  -4.54977933,  -3.54477569, -18.43401119, -20.22735899,
 -17.93166735,  -3.45367675,  -2.94392896,  -5.35330368,  -3.63605146,
  -4.221643,    -3.53575393,  -5.2765933,   -5.93633446,  -6.01068978,
  -9.851063,    -9.69890639, -10.90621383, -22.47993154, -18.27859025,
 -24.18460383], [-1.93219433,  -6.85041718,  -8.05721004,  -8.16034456, -12.28297663,
 -10.55698178, -12.54525463,  -9.81865817,  -7.76040726,  -8.37229773,
  30.59271806,  29.63999327,  31.75866242, -12.03921626,  -9.77970721,
 -13.7447929,   -6.43607781,  -6.48530213,  -6.40920553,  -6.58459117,
  -7.33965551,  -6.31137447,  -9.12714144,  -9.40647196,  -7.54607855,
  -9.26918163,  -8.09047784, -8.03193526, -10.47559131,  -7.09065454,
  -9.74600107], [-3.27480599,  -0.6503098,   -2.46774794,  -0.57219802,  -1.56732959,
  -1.59906133,  -1.54052922,  -1.9468644,   -0.98048774,  -2.87522601,
  -1.68186968,  -1.60500101,  -1.53165834,  15.48882665,  16.57552452,
  16.74196016,  -0.12644254,  -1.23211972,  -2.17160683,  -2.6884138,
  -1.5666072,   -0.68855349,  -2.16666206,  -1.13921385,  -2.07908944,
  -2.42583315,  -0.44228599,  -1.37433628,  -8.18082816,  -9.79283418,
  -8.66981066], [-1.86431597,  -6.22855296,  -4.90234987,  -5.02122885,  -8.13447361,
  -9.0762405,   -9.10691331,  -7.96705175,  -6.11934762,  -8.392208,
  -6.63332193,  -4.07208036,  -4.63504426, -10.71390305, -10.03632679,
 -10.10902894,  22.78354879,  22.90768731,  23.25797834,  -6.82373307,
  -7.3390698,   -6.10861279,  -7.25621662,  -5.14693406,  -7.11692998,
  -8.01944017,  -7.73232647,  -8.32869346, -12.00110694,  -7.91328712,
 -11.55815488], [-2.22522106,  -6.37222975,  -5.42910821,  -4.54318774,  -6.2655186,
  -7.4108343,   -6.20135824,  -6.98718542,  -6.04847834,  -7.31078885,
  -3.13117434,  -3.6099071,   -2.85838952,  -6.10783999,  -5.63452976,
  -6.039868,    -3.33128588,  -2.352829,    -3.33909242,  30.82487488,
  32.35953481,  30.84382264,  -5.14129718,  -5.88688988,  -6.65853972,
  -7.80688231,  -5.59613502,  -6.21699575,  -8.35584663,  -4.96880399,
  -5.2363686], [-2.12917822,  -5.65716041,  -5.57041,     -6.52384249,  -9.37255507,
  -8.85069677,  -9.63572163,  -5.9113801,   -7.11107829,  -7.09166644,
  -5.42775735,  -4.26226709,  -4.39819873,  -7.94264181,  -7.01746385,
  -7.7963714,   -3.36327788,  -5.09737468,  -6.38308768,  -4.69356155,
  -6.30455178,  -7.24355435,  33.65581914,  31.5088281,   31.91916611,
  -7.52522761,  -5.21123706,  -5.47917361,  -7.30650343,  -5.97841651,
  -7.56823442], [-2.74918361,  -7.18796015,  -6.32993169,  -6.23408559, -10.97698353,
 -11.2437675,  -10.19049364, -10.20340438, -10.36112328, -11.04239802,
  -5.71505069,  -4.93100099,  -4.48579552,  -4.26610076,  -3.94850867,
  -5.65045926,  -4.46804962,  -2.62201917,  -1.4885348,   -6.43568753,
  -7.68017299,  -7.2866343,   -5.66633931,  -5.48129944,  -5.58674934,
  48.55586826 , 47.57240458,  49.37585216,  -6.63260821,  -5.70345472,
  -7.34594327], [-5.0692268,   -0.26707292,   0.73340734,  -1.25601456,  -0.48458765,
  -1.20018483,  -0.44514898,  -2.05027612,  -3.75969519,  -3.01940099,
  -1.15002463,  -0.28361844,  -1.22249584,  -8.2924789,   -7.43060551,
  -8.61359746,  -1.1112762,   -1.02308268,   0.03809039,   1.02233534,
   0.10796672,  -0.84260307,  -0.65204213,  -1.61375545,   0.34609818,
  -0.49819958,   0.28484822,  -0.52174413,  18.82872148,  18.49730683,
  19.18324277]]


#validate(h, fingerprint, classes)

answer = raw_input("Want to predict for test data? (Y or N)\n")

if(answer == 'y' or answer == 'Y'):
	predict(h, classes, fingerprint)

"""
fig, ax = plt.subplots()
ax.imshow(fingerprint)
plt.show()
"""
