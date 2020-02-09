from keras.models import load_model
 # load model
model = load_model('saved_models/keras_cifar10_trained_model.h5')
print("Red neuronal cargada")
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
value = ""
while(True):
	# Capture frame-by-frame
	ret, img = cap.read()
	test = np.ndarray((1,32,32,3))
	# Our operations on the frame come here
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(img, 1.3, 5)

	for (x,y,w,h) in faces:
	    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0), cv2.FILLED) 
	    cv2.putText(img, value, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),2, lineType=cv2.LINE_AA)
	# Display the resulting frame
	#cv2.putText(img, value, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2, lineType=cv2.LINE_AA)
	cv2.imshow('imagen',img)
	k = cv2.waitKey(1) & 0xFF 

	if faces != ():
		img = cv2.resize(img, (32,32))
		test[0] = img
		pred = model.predict(test)
		if pred[0][0] > 0.5:
			print("Persona REAL ",pred)
			value = "Persona REAL"
		else:
			print("Persona FALSA ",pred)
			value = "Persona FALSA"
		cv2.imshow('imagen',img)
	if k == 27:
		break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
