from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image as sklimage
import feature_extraction, data_augmentation
import glob, pywt, csv, os.path, os, cv2


def get_train_features(fingerprint, classes):
	print "Finding features..."
	feat_dict = dict()

	for i in range(len(classes)): 
		features = []
		count_img = 0
		print "Getting images and calculating noise for "+classes[i]
		extensions = ('/*.jpg', '/*.JPG')
		images = os.listdir('dataset/train/'+classes[i])
		for ex in extensions:
			for image in glob.glob('dataset/train/'+classes[i]+ex): 
				#if count_img > 1:
				#	break
	
				print "image: ", count_img
				img = Image.open(image)

				if img.size[0] > img.size[1]:
					img = img.rotate(90)
			
				center = [img.size[0]/2, img.size[1]/2]
			
				img = img.crop((center[0]-256, center[1]-256, center[0]+256, center[1]+256))
				#data_augmentation.augment(img, images[count_img], classes[i])
				
				img = np.asarray(img)
					
				I_noise = feature_extraction.get_noise(img)
				I_noise = feature_extraction.increase_green_channel(I_noise)
				
				feat = calculate_features(I_noise)

				features.append(feat)
					
				count_img += 1
				
			for image in glob.glob('dataset/train/'+classes[i]+'/augmented/'+ex): 
				print "image: ", count_img
				
				img = Image.open(image)
				if img.size[0] > img.size[1]:
					img = img.rotate(90)	
				center = [img.size[0]/2, img.size[1]/2]
				img = img.crop((center[0]-256, center[1]-256, center[0]+256, center[1]+256))		
				img = np.asarray(img)
			
				I_noise = feature_extraction.get_noise(img)
				I_noise = feature_extraction.increase_green_channel(I_noise)
				
				feat = calculate_features(I_noise)

				features.append(feat)
					
				count_img += 1
		save_features(features, i)
					
	#return feat_dict


def get_fingerprint(classes):
	pattern_list = []
	for i in range(len(classes)):
		print "Getting fingerprint for "+classes[i]
		pattern_list.append(np.load('fingerprints/fingerprint_'+classes[i]+'.npy'))

	return pattern_list
	
def calculate_features(I_noise):	
	feat = []
	"""
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
	#statistical = feature_extraction.get_statistical_features(I_noise)
	
	#for stat in statistical:
	#	feat.append(stat)
	moments = feature_extraction.extract_features_moments(I_noise)
	for moment in moments:
		feat.append(moment)
	return feat
	
def save_features(feat, label):
	count = 1
	created = False
	while (not created):
		fn = 'moments_features'+str(count)+'.csv'
		if os.path.isfile(fn): 
			count += 1
		else:
			created = True
			
	with open(fn, 'a') as csvfile:
	
		for f in feat:
			f.append(label)
	
		for row in feat:
			writer = csv.writer(csvfile, delimiter = ',', quoting=csv.QUOTE_NONE)
			writer.writerow(row)

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
get_train_features(fingerprint, classes)



