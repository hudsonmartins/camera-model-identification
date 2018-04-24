from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from sklearn.feature_extraction import image as sklimage
from sklearn.linear_model import LogisticRegression
import feature_extraction
import glob, pywt, csv, os.path, os, cv2

def save_fingerprint(classes):
	pattern_list = []

	for i in range(len(classes)):
		noise_list = []
		count_img = 0   
		print "Getting fingerprint for "+classes[i]
		extensions = ('/*.jpg', '/*.JPG')
		for ex in extensions:
			for image in glob.glob('dataset/train/'+classes[i]+ex): 
				count_img += 1
				#if count_img > 50:
				#	break
		
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
					
		fingerprint = feature_extraction.get_pattern(noise_list)
		np.save('fingerprint_'+classes[i], fingerprint)
		
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
   	   
save_fingerprint(classes)
