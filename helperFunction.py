
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

    
cv2.rectangle(img, (545, 0), (1750, 618), (0, 0, 0), -1)
cv2.rectangle(img, (1374, 0), (1811, 795), (0, 0, 0), -1)

kernel = np.ones((7,7),np.uint8)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_bound = np.array([160,100,50])     
upper_bound = np.array([180,255,255])

mask = cv2.inRange(hsv, lower_bound, upper_bound)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

coordinates1x = []
coordinates1y = []
coordinates2x = []
coordinates2y = []

for i in contours:
    M = cv2.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
     
    if cx >750:
        coordinates2x.append(cx)
        coordinates2y.append(cy)
    if cx <=750:
        coordinates1x.append(cx)
        coordinates1y.append(cy)
    

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) 

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    return a, b

c, d = best_fit(coordinates1x, coordinates1y)
e, f = best_fit(coordinates2x, coordinates2y)

yfit1 = [c + d * xi for xi in coordinates1x]
yfit2 = [e + f * xi for xi in coordinates2x]

def necessaryCoordinates():
    return coordinates1x, yfit1, coordinates2x, yfit2

plt.plot(coordinates1x, yfit1, coordinates2x, yfit2, color="red", linewidth=1)
plt.axis('off')
img2 = cv2.imread('image2.jpg')
img2 = img2[:, :, ::-1]
plt.imshow(img2)
plt.savefig('plot.png',dpi=300, bbox_inches="tight", pad_inches=0.0)
plt.show()