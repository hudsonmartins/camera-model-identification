from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import feature_extraction, glob, csv, os.path, os

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
	statistical = feature_extraction.extract_features_moments(I_noise)
	
	for stat in statistical:
		feat.append(stat)
	return feat
	
def save_test_features(classes, fingerprint):
	created = False
	results = []
	count_image = 0
	for image in glob.glob('dataset/test/part4/*.tif'): 
		print "Image: ", count_image
		feat = []
		img = Image.open(image)
		img = np.asarray(img)
		
		#Getting noise image
		I_noise = feature_extraction.get_noise(img)
		I_noise = feature_extraction.increase_green_channel(I_noise)
		feat = calculate_features(I_noise)
		row = []
		row.append(image)
		for f in feat:
			row.append(f)
		count = 1
		while (not created):
			fn = 'stat_test_features'+str(count)+'.csv'
			if os.path.isfile(fn): 
				count += 1
			else:
				created = True
				
		with open(fn, 'a') as csvfile:
			writer = csv.writer(csvfile, delimiter = ',', quoting=csv.QUOTE_NONE)
			writer.writerow(row)
			
		count_image += 1
		
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
save_test_features(classes, fingerprint)
