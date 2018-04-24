from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import feature_extraction, glob, csv, os.path, os

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
		cross_corr = feature_extraction.get_cross_correlation(pattern, I_noise)
		for corr in cross_corr:
			feat.append(corr)
	"""
	statistical = feature_extraction.get_statistical_features(I_noise)
	
	for stat in statistical:
		feat.append(stat)
	"""
	return feat
	
def save_validation(fingerprint, classes):
	print "Saving..."
	targ = []
	count_cam = 0
	count_img = 0
	extensions = ('/*.jpg', '/*.JPG')
	for camera in classes:
		print "Calculating features for "+camera
		features = []
		n_acertos = 0
		for ex in extensions:
			for image in glob.glob('dataset/train/'+camera+'/validation/'+ex):
				print "Image: ", count_img
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
				features.append(feat)
				targ.append(count_cam)
				count_img += 1
		count = 1
		created = False
		while (not created):
			fn = 'validation_features'+str(count)+'.csv'
			if os.path.isfile(fn): 
				count += 1
			else:
				created = True
		
		with open(fn, 'a') as csvfile:

			for f in features:
				f.append(count_cam)

			for row in features:
				writer = csv.writer(csvfile, delimiter = ',', quoting=csv.QUOTE_NONE)
				writer.writerow(row)
				
		count_cam += 1
		
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
		
save_validation(fingerprint, classes)
