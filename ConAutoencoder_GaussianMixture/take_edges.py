from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

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

def take_edges(img):
    img = cv2.imread(os.path.join(abnormal_path,img_path))
    blur = cv2.GaussianBlur(img, (45, 45),0)#高斯模糊
    edges = cv2.Canny(blur, 1, 20)
    return edges#np.sum(edges==255)


if __name__ == '__main__':
    
    
    an_count=0
    abnormal_path = r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\1_abnormal\123"
    for img_name in os.listdir(abnormal_path):
        #print(i)
        img_path="{}/{}".format(abnormal_path,img_name)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.show()
        plt_show(take_edges(img_path))
        
        