#%%
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from model import ConvAutoencoder,ConvNetwork
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import ConvAutoencoder
from sklearn.mixture import GaussianMixture
from PIL import Image
import cv2

def take_edges(img):
    img = cv2.imread(os.path.join(normal_path,img_path))
    blur = cv2.GaussianBlur(img, (45, 45),0)#高斯模糊
    edges = cv2.Canny(blur, 1, 20)
    return np.sum(edges==255)
    
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

def detect_aoi(img_path,model):
    
    img = Image.open(img_path)
    im_tfs = transforms.Compose(
        [
        transforms.Resize(size=(re_size,re_size)),
        transforms.ToTensor(),
        ])
    
    img_tensor = im_tfs(img)
    batch = img_tensor.unsqueeze(0)
    #batch = batch.view(batch.size(0), -1)
    x = Variable(batch).to(device)
    dist = model.encode(x).cpu()
    
    #loss = mse_loss(xhat, x)
    #print(loss.item())
    return dist
#%%
if __name__ == '__main__':
    
    print(torch.cuda.get_device_properties(0))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    re_size=64
    z_dim = 32
    batch_size = 60
    
    AE = ConvAutoencoder(z_dim).to(device)
    AE.load_state_dict(torch.load("./old_flip_save_pth/483_AE_8.188586885808035e-05.pkl"))
    
    train_path=r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\train"
    
    train_data = datasets.ImageFolder(train_path,transform=transforms.Compose([
        transforms.Resize(size=(re_size,re_size)),#(h,w)
        #transforms.RandomHorizontalFlip(p=0.5),#依據p概率水平翻轉
        #transforms.RandomVerticalFlip(p=0.5),#依據p概率垂直翻轉
        transforms.ToTensor(),
    ]))

    print(train_data.classes)#获取标签
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0)
#%%
    features=torch.tensor([])
    with torch.no_grad():
        AE.eval()
        for step, (img,_) in enumerate(train_loader):
                
                input_tensor = Variable(img).to(device)
                output= AE.encode(input_tensor).cpu()
                features=torch.cat((features,output),0)
                print(step)
    
    gmm = GaussianMixture(n_components=1)
    gmm.fit(features.detach().numpy())
    OKscore = gmm.score_samples(features.detach().numpy())
    thred = OKscore.mean() - 4 * OKscore.std()
#%%
    with torch.no_grad():
        AE.eval()
        n_count=0
        normal_path = r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\0_normal\123"
        for img_name in os.listdir(normal_path):
            #print(i)
            img_path="{}/{}".format(normal_path,img_name)
            dist = detect_aoi(img_path,AE)
            ans = gmm.score_samples(dist)
            if(ans>thred):
                n_count+=1
    
#%%
    an_count=0
    abnormal_path = r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\1_abnormal\123"
    for img_name in os.listdir(abnormal_path):
        #print(i)
        img_path="{}/{}".format(abnormal_path,img_name)
        dist = detect_aoi(img_path,AE)
        ans = gmm.score_samples(dist.detach().numpy())
        edge_num = take_edges(img_path)
        if(ans<thred ):
            an_count+=1
        #elif(ans>thred and )
        else:
            img = Image.open(img_path)
            plt.imshow(img)
            plt.show()
            print(edge_num)
            
#%%
    print("normal:{}/{}".format(n_count,len(os.listdir(normal_path))))
            
    print("abnormal:{}/{}".format(an_count,len(os.listdir(abnormal_path))))
    
    print("正確率:{}%".format(100*(n_count+an_count)/(83+83)))
        #print(ans)
        #abnormal_total.append(torch.sum((dist - center) ** 2, dim=1))
#%%
    path_file=[r'D:\Lab702\AOI\ganomaly-master\data\AOI_part2\train\0.normal',
                r'D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\0_normal\123',
                r'D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\1_abnormal\123',]
    save_txt_img = []
    save_txt_val = []
    save_txt_class = []
    with torch.no_grad():
        AE.eval()
        n_count=0

        for i,img_path in enumerate(path_file):
                for img_name in os.listdir(img_path):
                    #print(img_name)
                    input_img_path="{}/{}".format(img_path,img_name)
                    dist = detect_aoi(input_img_path,AE)
                    ans = gmm.score_samples(dist.detach().numpy())
                    save_txt_img.append(img_name)
                    save_txt_val.append(float(ans-thred))
                    if(i>1):
                        save_txt_class.append(1)
                    else:
                        save_txt_class.append(0)

    save_txt_val = np.array(save_txt_val)
    max_,min_ = save_txt_val.max() ,save_txt_val.min()
    new_save_txt_val = (save_txt_val-min_)/(max_-min_)
    
    path = 'output_data_2.txt'
    with open(path, 'w') as f:
        for i,img_name in enumerate(save_txt_img):
            
            write = f'{img_name} {new_save_txt_val[i]} {save_txt_class[i]}'
            print(write)
            f.write(f'{write}\n')
        
    
    pass







