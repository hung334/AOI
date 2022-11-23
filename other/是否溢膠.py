import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
import time
from PIL import ImageEnhance





def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    #cv.imshow("custom_blur_demo", dst)
    return dst

def contrast_img(img1, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img1.shape

    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    
    return dst
    #cv2.imshow('original_img', img)
    #cv2.imshow("contrast_img", dst)

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
    

    
    img = cv2.imread("center/no_1.png")
    pil_img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
    plt_show(img)
    blur = cv2.GaussianBlur(img, (45, 45),0)#高斯模糊
    plt_show(blur)
    edges = cv2.Canny(blur, 1, 20)
    plt_show(edges)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    total_abnormal=[]
    abnormal_path = r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\1_abnormal\123"
    for img_path in os.listdir(abnormal_path):
        img = cv2.imread(os.path.join(abnormal_path,img_path))
        blur = cv2.GaussianBlur(img, (45, 45),0)#高斯模糊
        edges = cv2.Canny(blur, 1, 20)
        print(np.sum(edges==255))
        if(np.sum(edges==255)<=254):
            plt_show(img)
            plt_show(edges)
        total_abnormal.append(np.sum(edges==255))
    total_abnormal=np.array(total_abnormal)


    total_normal=[]
    normal_path = r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\0_normal\123"
    for img_path in os.listdir(normal_path):
        img = cv2.imread(os.path.join(normal_path,img_path))
        blur = cv2.GaussianBlur(img, (45, 45),0)#高斯模糊
        edges = cv2.Canny(blur, 1, 20)
        print(np.sum(edges==255))
        # if(np.sum(edges==255)>=254):
        #     plt_show(img)
        #     plt_show(edges)
        total_normal.append(np.sum(edges==255))
    total_normal=np.array(total_normal)
    
    
    print("abnormal-max:{}".format(np.max(total_abnormal)))
    print("abnormal-min:{}".format(np.min(total_abnormal)))
    print("abnormal-mean:{}".format(np.mean(total_abnormal)))
    print("normal-max:{}".format(np.max(total_normal)))
    print("normal-min:{}".format(np.min(total_normal)))
    print("normal-mean:{}".format(np.mean(total_normal)))
    # # 增强亮度
    # enh_bri = ImageEnhance.Brightness(pil_img)
    # brightness = 1.1
    # image_brightened = enh_bri.enhance(brightness)
    # plt_show(cv2.cvtColor(np.asarray(image_brightened),cv2.COLOR_RGB2BGR))
    
    # # 色度增强
    # enh_col = ImageEnhance.Color(image_brightened)
    # color = 5
    # image_colored = enh_col.enhance(color)
    # plt_show(cv2.cvtColor(np.asarray(image_colored),cv2.COLOR_RGB2BGR))
    
    # # 对比度增强
    # enh_con = ImageEnhance.Contrast(image_brightened)
    # contrast = 2.5
    # image_contrasted = enh_con.enhance(contrast)
    # plt_show(cv2.cvtColor(np.asarray(image_contrasted),cv2.COLOR_RGB2BGR))
    
    
    # cc_img = contrast_img(img, 1, 15)
    # plt_show(cc_img)
    
    # cc_gray = cv2.cvtColor(cc_img, cv2.COLOR_BGR2GRAY)
    # plt_show(cc_gray)
    # plt_show(gray)
    
    #dst = cv2.equalizeHist(cc_img)
    #plt_show(dst)
    
    
    
    #ret, thresh1 = cv2.threshold(dst, 30, 255, 0)
    #plt_show(thresh1)
    
    
    # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # plt.hist(gray.ravel(), 256, [0, 256])
    # plt.show()
    
    
    # blockSize = 15
    # C_val = 5
    
    # #自适应阈值分割
    # img_ada_mean=cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize,C_val)
    # img_ada_gaussian=cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize,C_val)
    
    # plt_show(img_ada_mean)
    # plt_show(img_ada_gaussian)
    
    
