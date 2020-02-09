import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0
z = 0
while(True):
	# Capture frame-by-frame
	ret, img = cap.read()

	# Our operations on the frame come here
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(img, 1.3, 5)

	for (x,y,w,h) in faces:
	    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0), cv2.FILLED) 
	# Display the resulting frame
	cv2.imshow('imagen',img)
	k = cv2.waitKey(1) & 0xFF 

	if faces != ():
		if k == 115 or k == 83:
			print("guardar")
			count += 1
			img2 = img[y-z:y+int(h/3),x-int(z/3):x+w+int(z/3)]
			img2=cv2.resize(img2,(96,96))
			cv2.imwrite("dataset" + '/' + str(count) + ".jpg", img)
	if k == 27:
		break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
