import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
import time


def plt_show(image):
    
    if(len(image.shape)==3):
        image = image[:,:,::-1]
    elif(len(image.shape)==2):
        image = image
    
    plt.imshow(image,cmap ='gray')
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()
    
    
    
if __name__ == '__main__':
    
    crop_after_file="after_crop"
    if not os.path.isdir(crop_after_file):
        os.mkdir(crop_after_file)
    
    crop_img = cv2.imread("62.png")
    template = cv2.imread("datasets/04_OrgA.png")
    
    
    draw_img = crop_img.copy()
    white = np.zeros(template.shape, np.uint8)   
    white.fill(255)
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    ''' draw_all'''
    #cv2.drawContours(draw_img,contours,-1,(255,0,255),1)
    #plt_show(draw_img)
    
    
    for i,c in enumerate(contours):
        
        (x, y, w, h) = cv2.boundingRect(c)
        #print(x, y, w, h)
        if( w>10 and h>10):
            #if(w>300 and h>300):
            #    cv2.imwrite("center.png",crop_img[y:y+h,x:x+w])
            cv2.imwrite("{}/{}.png".format(crop_after_file,i),crop_img[y:y+h,x:x+w])
            cv2.rectangle(white, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #plt_show(crop_img[y:y+h,x:x+w])
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    
    plt_show(template)
    plt_show(white)
    plt_show(draw_img)
    