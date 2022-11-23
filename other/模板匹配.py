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




if __name__ == '__main__':
    
    img_rgb = cv2.imread("datasets/03_OrgA/~VC_R_OrgA_0_0_105514_.png")#("test00.png"),04_OrgA_0_4_103213_.png
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    draw_img = img_rgb.copy()
    #reg_img_gray = img_gray.copy()
    
    template = cv2.imread("datasets/03_OrgA.png",0)#('test0 (2).png',0)
    w, h = template.shape[::-1]
    
    
    fig = plt.figure()
    draw_imgs=[]
    
    gif_file1="gif"
    if not os.path.isdir(gif_file1):
        os.mkdir(gif_file1)
    
    save_file1="draw_imgs"
    if not os.path.isdir(os.path.join(gif_file1,save_file1)):
        os.mkdir(os.path.join(gif_file1,save_file1))
    
    save_file2="img_gray"
    if not os.path.isdir(os.path.join(gif_file1,save_file2)):
        os.mkdir(os.path.join(gif_file1,save_file2))
    
    save_file3="crop"
    if not os.path.isdir(os.path.join(gif_file1,save_file3)):
        os.mkdir(os.path.join(gif_file1,save_file3))
    

    t1 =time.time()
    while(1):
        

        
        
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.75
        #res大于70%
        loc = np.where( res >= threshold)
        
        for pt in zip(*loc[::-1]):
            #print(pt)
            
            cv2.rectangle(draw_img, pt, (pt[0] + w, pt[1] + h), (7,249,151), 10)
            img_gray[pt[1]:pt[1] + h,pt[0]:pt[0]+w] = 255
            plt_show(draw_img)
            plt_show(img_gray)
            plt_show(img_rgb[pt[1]:pt[1] + h,pt[0]:pt[0]+w])
            
            #cv2.imwrite("gif/draw_imgs/{}.png".format(len(os.listdir("gif/draw_imgs"))),draw_img)
            #cv2.imwrite("gif/img_gray/{}.png".format(len(os.listdir("gif/img_gray"))),img_gray)
            #cv2.imwrite("gif/crop/{}.png".format(len(os.listdir("gif/crop"))),img_rgb[pt[1]:pt[1] + h,pt[0]:pt[0]+w])
            
            break 
        
        
        ret, thresh1 = cv2.threshold(img_gray, 127, 255, 0)
        
        total_nmu = thresh1.shape[0]*thresh1.shape[1]
        black_num = np.sum(thresh1==0)

        
        print((black_num/total_nmu))
        if((black_num/total_nmu)<0.025):
            break
    t2 =time.time()
    
    print(t2-t1)
