import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, datasets, models
import torchvision
import pylab
import matplotlib.pyplot as plt
from Net import *
from PIL import Image


def histogram(normal,abnormal,savename=""):

    n1, _, _ = plt.hist(normal, bins=len(normal), alpha=0.5, label='Normal')
    n2, _, _ = plt.hist(abnormal, bins=len(abnormal), alpha=0.5, label='Abnormal')
    h_inter = np.sum(np.minimum(n1, n2)) / np.sum(n1)
    plt.xlabel("MSE")
    plt.ylabel("Number of Data")
    xmax = max(normal.max(), abnormal.max())
    plt.xlim(0, xmax)
    plt.text(x=xmax*0.01, y=max(n1.max(), n2.max()), s="Histogram Intersection: %.3f" %(h_inter))
    plt.legend(loc='upper right')
    #plt.savefig(savename)
    plt.show()
    plt.close()

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
    batch = batch.view(batch.size(0), -1)
    x = Variable(batch).to(device)
    
    xhat = model(x).to(device)
    
    loss = mse_loss(xhat, x)
    print(loss.item())
    return loss.item()
    #label_pred = outputs.max(1)[1].squeeze().cpu().data.numpy()
    #plt.imshow(img)
    









if __name__ == '__main__':
    

    print(torch.cuda.get_device_properties(0))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    re_size=224
    z_dim = 64
    batch_size = 1
    
    model =Autoencoder(z_dim,re_size).to(device)
    #ConvAutoencoder().to(device)#Autoencoder(z_dim,re_size).to(device)
    model.load_state_dict(torch.load('224_save/best_2840_0.0001145708083640784.pkl'))
    model.eval()
    
    mse_loss = nn.MSELoss().to(device)
    
    train_path=r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\train"
    test_path=r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\0_normal"  #0_normal,1_abnormal
    #dev_path="./datasets/unet_crop_Dev"
    
    #img_path = r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\train\0.normal\03_6x6_OrgA_0_3_152639__0.png"
    #detect_aoi(img_path,model)
    
    normal_total=[]
    abnormal_total=[]
    total=[]
    
    normal_path = r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\0_normal\123"
    
    for img_name in os.listdir(normal_path):
        #print(i)
        img_path="{}/{}".format(normal_path,img_name)
        output = detect_aoi(img_path,model)
        normal_total.append(output)
        total.append(output)
    
    abnormal_path = r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\1_abnormal\123"
    
    for img_name in os.listdir(abnormal_path):
        #print(i)
        img_path="{}/{}".format(abnormal_path,img_name)
        output = detect_aoi(img_path,model)
        abnormal_total.append(output)
        total.append(output)
        if(output<=0.0004):
            img = Image.open(img_path)
            plt.imshow(img)
            plt.show()
            
    normal_total = np.array(normal_total)
    abnormal_total = np.array(abnormal_total)
    
    #histogram(total[:len(os.listdir(normal_path))],total[len(os.listdir(normal_path)):],savename="")
    
    
    
    #plt.hist([normal_total,abnormal_total], bins = np.linspace(-1, 1, 83*2), label=['normal','abnormal'])
    
    bins = 5#np.linspace(-10, 10, 5)
    
    #plt.hist(total, 100, label='normal')
    plt.hist(normal_total, 5, alpha = 0.5, label='normal')
    plt.hist(abnormal_total,100, alpha = 0.5, label='abnormal')
    #plt.hist(total,100, alpha = 0.5, label='abnormal')
    
    #plt.bar(normal_total, [5 for i in range(83)], label='normal', width=0.000005) 
    #plt.bar(abnormal_total, [5 for i in range(83)], label='abnormal', width=0.000005)
    plt.xlabel("MSE")
    plt.ylabel("Number of Data")
    plt.xlim(0, 0.00075)
    plt.legend(loc='upper left')
    plt.show()
    
    
    a = np.array(normal_total)
    print("max:",np.max(a))
    
    Correct_rate = 0
    n_Correct_rate = 0
    ab_Correct_rate = 0
    
    boundary_line = 0.00033#0.00041、0.000382
    for data in normal_total:
        if(data<=boundary_line):
            n_Correct_rate+=1
            Correct_rate+=1
    for data in abnormal_total:
        if(data>boundary_line):
            ab_Correct_rate+=1
            Correct_rate+=1
    
    print("normal:{}/{}".format(n_Correct_rate,len(normal_total)))         
    print("abnormal:{}/{}".format(ab_Correct_rate,len(abnormal_total)))
    print(Correct_rate/(len(normal_total)+len(abnormal_total))*100,'%')
#%%

    path_file=[r'D:\Lab702\AOI\ganomaly-master\data\AOI_part2\train\0.normal',
                r'D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\0_normal\123',
                r'D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\1_abnormal\123',]
    save_txt_img = []
    save_txt_val = []
    save_txt_class = []
    with torch.no_grad():
        model.eval()
        n_count=0

        for i,img_path in enumerate(path_file):
                for img_name in os.listdir(img_path):
                    #print(img_name)
                    input_img_path="{}/{}".format(img_path,img_name)
                    output = detect_aoi(input_img_path,model)
                    save_txt_img.append(img_name)
                    save_txt_val.append(float(output-boundary_line))
                    if(i>1):
                        save_txt_class.append(1)
                    else:
                        save_txt_class.append(0)

    save_txt_val = np.array(save_txt_val)
    max_,min_ = save_txt_val.max() ,save_txt_val.min()
    new_save_txt_val = (save_txt_val-min_)/(max_-min_)
    
    path = 'output_data_1.txt'
    with open(path, 'w') as f:
        for i,img_name in enumerate(save_txt_img):
            
            write = f'{img_name} {new_save_txt_val[i]} {save_txt_class[i]}'
            print(write)
            f.write(f'{write}\n')
        
    
    pass




#%%
    '''
    test_data = datasets.ImageFolder(test_path,transform=transforms.Compose([
        transforms.Resize(size=(re_size,re_size)),#(h,w)
        #transforms.RandomCrop(size=(256,256), padding=5),
        #transforms.ColorJitter(brightness=0.2, contrast=0.5,saturation=0.5),
        #transforms.RandomHorizontalFlip(p=0.5),#依據p概率水平翻轉
        #transforms.RandomVerticalFlip(p=0.5),#依據p概率垂直翻轉
        #transforms.RandomRotation((-45,45)),#隨機角度旋轉
        #transforms.RandomGrayscale(p=0.4),
        transforms.ToTensor()
    ]))
    
    #print(train_data.classes)#获取标签

    test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True,num_workers=0)
    
    with torch.no_grad():
        model.eval()
        test_loss_reg = 0
        for img,_ in test_loader:

            
            x = img.view(img.size(0), -1)
            x = Variable(x).to(device)
            
            xhat = model(x).to(device)
            
            plt_xhat = img.view(x.shape[0],3,re_size,re_size)
            grid_img = torchvision.utils.make_grid(plt_xhat.cpu().data, nrow=10)
            #plt.imshow(np.uint8(grid_img.permute(1, 2, 0)*255))
            #plt.show()
            
            #plt_xhat = xhat.view(x.shape[0],3,re_size,re_size)
            #grid_img = torchvision.utils.make_grid(plt_xhat.cpu().data, nrow=10)
            #plt.imshow(np.uint8(grid_img.permute(1, 2, 0)*255))
            #plt.show()
            
            # 出力画像（再構成画像）と入力画像の間でlossを計算
            loss = mse_loss(xhat, x)
            #losses[epoch] = losses[epoch] * (i / (i + 1.)) + loss * (1. / (i + 1.))
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            #i += 1
            test_loss_reg +=loss.cpu().data
            
            print('loss: {:.4f}'.format(loss))
        
        
        avg_test_loss = test_loss_reg/len(test_loader)
        #if(avg_train_loss<=best_low_loss):
        #    torch.save(model.state_dict(), '{}/best_{}.pkl'.format(save_file,epoch+1,))
        print("*"*10,avg_test_loss)
        torch.cuda.empty_cache()
    '''