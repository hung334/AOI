"""
21.4.3-形状匹配.py:
函数 cv2.matchShape() 可以帮我们比 两个形状或 廓的相似度。
如果返回值越小， 匹配越好。它是根据 Hu 矩来计算的。文档中对不同的方法有解释。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def plt_show(image):
    
    if(len(image.shape)==3):
        image = image[:,:,::-1]
    elif(len(image.shape)==2):
        image = image
    
    plt.imshow(image,cmap ='gray')
    plt.xticks([])
    plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()


if __name__ == '__main__':
    
    img1 = cv2.imread('test/04_7x7_OrgA_0_0_103213_.png')
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) 
    
    img2 = cv2.imread('test/04_pattern.png')
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
    
    ret, thresh1 = cv2.threshold(gray1, 127, 255, 0)
    ret, thresh2 = cv2.threshold(gray2, 127, 255, 0)
    
    
    contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(thresh2, 2, 1)
    # cnt1 = contours[0]

    cnt2 = contours2[0]
    
    cv2.drawContours(img1,contours1,-1,(0,0,255),3)
    plt_show(img1)
    
    cv2.drawContours(img2,contours2[2],-1,(0,0,255),1)
    plt_show(img2)
    
    # for contour in contours1:
    #     ret = cv2.matchShapes(contour, cnt2, 1, 0.0)
    #     print(ret)
    #     if(ret<0.1):
    #             cv2.drawContours(img1,contour,-1,(0,0,255),3)
    # plt_show(img1)
    
    
    # #cv2.drawContours(img2,contours2[0],-1,(0,0,255),3)
    # plt_show(img2)
    

    
    
    #cv2.imshow("img", img1)  
    #cv2.waitKey(0) 