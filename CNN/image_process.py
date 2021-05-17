import cv2
 
img = cv2.imread(r'C:\Users\83621\Documents\vscode\machineLearning\CNN\DSCF7691.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
 
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,0,255),3)
 
cv2.imshow("img", img)
cv2.waitKey(0)
