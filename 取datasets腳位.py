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
    plt.xticks([])
    plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()
    

def crop_out_center(crop_img,template):
    
    draw_img = crop_img.copy()
    white = np.zeros(template.shape, np.uint8)   
    white.fill(255)
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    for i,c in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        #print(x, y, w, h)
        #if( w>10 and h>10):
        if(w>100 and h>100):
            cv2.rectangle(white, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            plt_show(draw_img)
            return crop_img[y:y+h,x:x+w]

def template_matching2(image,template,save_path):
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h,c= template.shape
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    count = 0
    
    while(1):
        
        res = cv2.matchTemplate(img_gray,gray_template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.75
        #res大于70%
        loc = np.where( res >= threshold)
        crop_img = None
        for pt in zip(*loc[::-1]):
            #print(pt)
            
            #cv2.rectangle(draw_img, pt, (pt[0] + w, pt[1] + h), (7,249,151), 10)
            img_gray[pt[1]:pt[1] + h,pt[0]:pt[0]+w] = 255
            crop_img = image[pt[1]:pt[1] + h,pt[0]:pt[0]+w]
            #plt_show(draw_img)
            plt_show(img_gray)
            #plt_show(img_rgb[pt[1]:pt[1] + h,pt[0]:pt[0]+w])
            break
        center_crop = crop_out_center(crop_img,crop_img)
        #cv2.imwrite("{}_{}.png".format(save_path,count),center_crop)
        count+=1
        #####
        ret, thresh1 = cv2.threshold(img_gray, 127, 255, 0)
        total_nmu = thresh1.shape[0]*thresh1.shape[1]
        black_num = np.sum(thresh1==0)
        print((black_num/total_nmu))
        if((black_num/total_nmu)<0.025):
            break

def template_matching(image,template,save_path):
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h,c= template.shape
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    count = 0
    
    while(1):
        
        res = cv2.matchTemplate(img_gray,gray_template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left=max_loc
        #cv2.rectangle(draw_img,top_left,(top_left[0]+w,top_left[1]+h),(0,0,225),5)
        #plt_show(draw_img)
        img_gray[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w] = 255
        crop_img = image[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w]#裁切腳位
        plt_show(img_gray)
        plt_show(crop_img)
        #center_crop = crop_out_center(crop_img,crop_img)#裁切中間
        #cv2.imwrite("{}_{}.png".format(save_path,count),center_crop)
        cv2.imwrite("{}_{}.png".format(save_path,count),crop_img)
        count+=1
        #####
        ret, thresh1 = cv2.threshold(img_gray, 127, 255, 0)
        total_nmu = thresh1.shape[0]*thresh1.shape[1]
        black_num = np.sum(thresh1==0)
        print((black_num/total_nmu))
        if((black_num/total_nmu)<0.025):
            break
        
if __name__ == '__main__':
    
    
    img_name = "~04_OrgA_0_4_103213_"
    img_template = "04_OrgA"
    
    save_path = "Foot_position_datasets/{}".format(img_name)
    image = cv2.imread("datasets/{}/{}.png".format(img_template,img_name))
    template = cv2.imread("datasets/{}.png".format(img_template))
    template_matching(image,template,save_path)