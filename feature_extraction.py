import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from scipy.stats import moment
import pywt, cv2


def increase_green_channel(I_noise):
	for row in I_noise:
		for pixel in row:
			pixel[0] = pixel[0]*0.3
			pixel[1] = pixel[1]*0.6
			pixel[2] = pixel[2]*0.1
	return I_noise

def get_fingerprint(noises, images):

	add_prod = np.zeros((len(images[0]), len(images[0][0]), 3))
	add_sqd = np.zeros((len(images[0]), len(images[0][0]), 3))

	for i in range(len(images)):
		add_prod = np.add(add_prod, np.multiply(images[i], noises[i]))
		add_sqd = np.add(add_sqd, np.square(images[i]))
		
	fingerprint = np.divide(add_prod, add_sqd)
	
	return fingerprint
						
def extract_features(fingerprint):
	print statistical_moments(fingerprint)

def statistical_moments(fingerprint):
	#first 3 central normalized moments nu20, nu11, nu02
	b, g, r = cv2.split(fingerprint)
	
	r_moments = cv2.moments(r)
	g_moments = cv2.moments(g)
	b_moments = cv2.moments(b)
	
	features = [[r_moments['nu20'], r_moments['nu11'], r_moments['nu02']], [g_moments['nu20'], g_moments['nu11'], g_moments['nu02']], [b_moments['nu20'], b_moments['nu11'], b_moments['nu02']]]
	
	return features
	
def extract_features_moments(I_noise):	
	"""
		Param: Noise image
		Returns: 63 Features
	"""
	b, g, r = cv2.split(I_noise)
	
	'''
	fig, ax = plt.subplots(ncols = 3)
	ax[0].imshow(r)
	ax[1].imshow(g)
	ax[2].imshow(b)
	plt.show()
	'''

	#Single wavelet decomposition for each channel
	RH, RV, RD = pywt.dwt2(r, 'db8')[1]
	GH, GV, GD = pywt.dwt2(g, 'db8')[1]
	BH, BV, BD = pywt.dwt2(b, 'db8')[1]
	wv_components = [RH, RV, RD, GH, GV, GD, BH, BV, BD]
	moments = []
	
	for component in wv_components:
		moment = cv2.moments(component)
		#for k in range(1,10):
		#	moments.append(moment(component, moment=k))
		moments.append(moment['mu20'])
		moments.append(moment['mu11'])
		moments.append(moment['mu02'])
		moments.append(moment['mu30'])
		moments.append(moment['mu21'])
		moments.append(moment['mu12'])
		moments.append(moment['mu03'])
	#print len(moments)

	return moments
	
def get_noise(img):
	wv_denoise = denoise_wavelet(img, wavelet='db8', multichannel=True, wavelet_levels = 4)
	noise = img - wv_denoise
	return noise, wv_denoise
