import numpy as np
import cv2
import sys
import math
import numpy

def gaussian(x,sigma):
    return (1.0/(2*numpy.pi*(sigma**2)))*numpy.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return numpy.sqrt(numpy.abs((x1-x2)**2+(y1-y2)**2))

def ApplyBilateralFilter(source, filtered_image, x, y, diameter, sigma_r, sigma_s):
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[int(neighbour_x)][int(neighbour_y)] - source[x][y], sigma_r)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi*gs
            i_filtered += source[int(neighbour_x)][int(neighbour_y)] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered // Wp
    filtered_image[x][y] = i_filtered

def ApplyGeodezicFilter(source, filtered_image, x, y, diameter, sigma_r, sigma_s):
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        minim = 300
        while j < diameter:
            neighbour_x = x - hl + i
            neighbour_y = y - hl + j
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            if(minim < (source[int(neighbour_x)][int(neighbour_y)] - source[x][y])).all():    
               minim = source[int(neighbour_x)][int(neighbour_y)] - source[x][y]
            w =  gaussian(minim , sigma_r)
            i_filtered += source[int(neighbour_x)][int(neighbour_y)] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered // Wp
    filtered_image[x][y] = i_filtered

def padding(im,r):
    return np.pad(im,((r, r),(r, r)), 'reflect')
    
def ApplyGuidedFilter(im,guide,r,epsilone):
    im = np.array(im, np.float32)
    a = np.zeros((im.shape[0],im.shape[1]), np.float32)
    b = np.zeros((im.shape[0],im.shape[1]), np.float32)
    O = np.array(im, np.float32, copy=True)  
    n=np.shape(im)[0]
    m=np.shape(im)[1] 
    a_k = np.zeros((n,m), np.float32)
    b_k = np.zeros((n,m), np.float32)
    w=2*r+1
    for i in range(r,n-r):
        for j in range(r,m-r):
            I=guide[i-r:i+r+1 ,j-r:j+r+1 ]
            P=im[i-r:i+r+1 ,j-r:j+r+1 ]
            mu_k = np.mean(I)
            delta_k = np.var(I)
            P_k_bar = np.mean(P)
            somme = np.dot(np.ndarray.flatten(I), np.ndarray.flatten(P))/(w**2)
            a_k[i,j] = (somme - mu_k * P_k_bar) / (delta_k + epsilone)
            b_k[i,j] = P_k_bar - a_k[i,j] * mu_k   
    a=a_k[r:n-r+1,r:m-r+1]
    b=b_k[r:n-r+1,r:m-r+1]
    a=padding(a,r)
    b=padding(b,r)
    for i in range(r, n-r):
        for j in range(r, m-r):
            a_k_bar = a[i-r : i+r+1, j-r : j+r+1].sum()/(w*w)
            b_k_bar = b[i-r : i+r+1, j-r : j+r+1].sum()/(w*w)
            O[i-r,j-r] = a_k_bar * guide[i,j] + b_k_bar
    return O
    
def bilateral(source, filter_diameter, sigma_r, sigma_s):
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            ApplyBilateralFilter(source, filtered_image, i, j, filter_diameter, sigma_r, sigma_s)
            j += 1
        i += 1
    return filtered_image

def geodezic(source, filter_diameter, sigma_r, sigma_s):
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            ApplyGeodezicFilter(source, filtered_image, i, j, filter_diameter, sigma_r, sigma_s)
            j += 1
        i += 1
    return filtered_image

if __name__ == "__main__":
    print("Alege filtrul pe care vrei sa il folosesti")
    print("1.Bilateral FIltering")
    print("2.Geodezic FIltering")
    print("3.Guided FIlter")
    n = int(input()) 
    #you must change this
    src = cv2.imread("/home/razvan/Desktop/photo.jpg") 
    if(n == 1)  :
       filtered_image = bilateral(src, 4,16,12)
       cv2.imwrite("bilateral.png", filtered_image)
    elif(n==2) :
       filtered_image = geodezic(src, 4,16,12)
       cv2.imwrite("geodezic.png", filtered_image)
    elif(n==3):
       guide = cv2.imread("/home/razvan/Desktop/verm.jpg")
       filtered_image = ApplyGuidedFilter(src,guide,5,4)
       cv2.imwrite("guided.png", filtered_image)
    else :
       print("Optiune incorecta")
