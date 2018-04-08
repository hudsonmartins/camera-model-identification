from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
import glob, pywt, feature_extraction


img_list = []
noise_list = []
denoised_list = []

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
   	   
print "Getting images and calculating noise..."

count_img = 0   	   
for image in glob.glob('dataset/train/'+classes[0]+'/*.jpg'): 
	count_img += 1
	if count_img > 1:
		break
		
	print count_img
	img = Image.open(image).convert('RGB')

	if img.size[0] > img.size[1]:
		img = img.rotate(90, expand=True)
	
	img = np.asarray(img)/255.0
	I_noise, denoised = feature_extraction.get_noise(img)
	
	img_list.append(img)
	noise_list.append(I_noise)
	
print "Calculating Fingerprint for the model "+classes[0]
fingerprint = feature_extraction.get_fingerprint(noise_list, img_list)
feature_extraction.extract_features(fingerprint)

"""
fig, ax = plt.subplots()
ax.imshow(fingerprint)
plt.show()
"""
