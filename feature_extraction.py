import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from scipy.stats import moment
from scipy.signal import correlate2d
import pywt, cv2


def increase_green_channel(I_noise):
	b,g,r = cv2.split(I_noise)
	b = b*0.1
	g = g*0.3
	r = r*0.3
	I_noise = cv2.merge((b,g,r))

	return I_noise

def mean(v):
	return sum(v)/len(v)

def covariance(X, Y):
	return np.dot(X - mean(X), Y - mean(Y))

def correlation(X, Y):
	X = X.flatten()
	Y = Y.flatten()
	return covariance(X,Y)/(np.sqrt(covariance(X,X) * covariance(Y,Y))+(0.0000001))

def get_correlation(pattern, Inoise):
	b_pattern, g_pattern, r_pattern = cv2.split(pattern)
	b_noise, g_noise, r_noise = cv2.split(Inoise)

	r_pattern = r_pattern.flatten()
	r_noise = r_noise.flatten()
	
	g_pattern = g_pattern.flatten()
	g_noise = g_noise.flatten()
	
	b_pattern = b_pattern.flatten()
	b_noise = b_noise.flatten()
	
	r = correlation(r_pattern, r_noise)
	g = correlation(g_pattern, g_noise)
	b = correlation(b_pattern, b_noise)
	
	return r, g, b
	
def get_cross_correlation(pattern, Inoise):
	b_pattern, g_pattern, r_pattern = cv2.split(pattern)
	b_noise, g_noise, r_noise = cv2.split(Inoise)

	r_pattern = r_pattern.flatten()
	r_noise = r_noise.flatten()
	
	g_pattern = g_pattern.flatten()
	g_noise = g_noise.flatten()
	
	b_pattern = b_pattern.flatten()
	b_noise = b_noise.flatten()
	
	r1 = correlation(r_pattern, g_noise)
	r2 = correlation(r_pattern, b_noise)
	
	g1 = correlation(g_pattern, b_noise)
	g2 = correlation(g_pattern, r_noise)
	
	b1 = correlation(b_pattern, r_noise)
	b2 = correlation(b_pattern, g_noise)
	
	return [r1, r2, g1, g2, b1, b2]

def get_statistical_features(I_noise):
	b, g, r = cv2.split(I_noise)
	
	r_mean = np.mean(r.flatten())
	g_mean = np.mean(g.flatten())
	b_mean = np.mean(b.flatten())
	
	r_var = np.var(r.flatten())
	g_var = np.var(g.flatten())
	b_var = np.var(b.flatten())
	
	r_std =	np.std(r.flatten())
	g_std = np.std(g.flatten())
	b_std = np.std(b.flatten())
	
	return [r_mean, g_mean, b_mean, r_var, g_var, b_var, r_std, g_std, b_std]
	
	
def get_pattern(noises):
	sum_noises = np.zeros((len(noises[0]), len(noises[0]), 3))
	
	for i in range(len(noises)):
		sum_noises = np.add(sum_noises, noises[i])
	
	return sum_noises/len(noises)


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

def statistical_moments(Inoise):
	#first 3 central normalized moments nu20, nu11, nu02
	b, g, r = cv2.split(Inoise)
	
	r_moments = cv2.moments(r)
	g_moments = cv2.moments(g)
	b_moments = cv2.moments(b)
	
	features = [r_moments['nu20'], r_moments['nu11'], r_moments['nu02'], g_moments['nu20'], g_moments['nu11'], g_moments['nu02'], b_moments['nu20'], b_moments['nu11'], b_moments['nu02']]
	
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
		#for k in range(1,10):
		#	moments.append(moment(component.ravel(), moment=k))
		moment = cv2.moments(component)
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
	noise = img - wv_denoise*255.0
	
	return noise
