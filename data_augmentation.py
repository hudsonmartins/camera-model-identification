import cv2, os, numpy as np
from PIL import Image
from skimage.exposure import adjust_gamma
def augment(img, fname, camera):
	if fname != 'validation' and fname != 'augmented':
		directory = "dataset/train/"+camera+"/augmented/"
		if not os.path.exists(directory):
			os.mkdir(directory)		

		img.save(directory+"compressed70_"+fname, "jpeg", quality=70)
		img.save(directory+"compressed90_"+fname, "jpeg", quality=90)
		img = np.asarray(img) 
		cv2.imwrite(directory+"resize0.5_"+fname, cv2.resize(img,None, fx= 0.5, fy=0.5, interpolation = cv2.INTER_CUBIC))
		cv2.imwrite(directory+"resize0.8_"+fname, cv2.resize(img,None, fx= 0.8, fy=0.8, interpolation = cv2.INTER_CUBIC))
		cv2.imwrite(directory+"resize1.5_"+fname, cv2.resize(img,None, fx= 1.5, fy=1.5, interpolation = cv2.INTER_CUBIC))
		cv2.imwrite(directory+"resize2_"+fname, cv2.resize(img, None, fx= 2, fy=2, interpolation = cv2.INTER_CUBIC))
		cv2.imwrite(directory+"gamma0.8_"+fname, adjust_gamma(img, gamma=0.8))
		cv2.imwrite(directory+"gamma1.2_"+fname, adjust_gamma(img, gamma=1.2))


