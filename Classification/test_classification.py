import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from AE_model import ConvNetwork_classification
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter       

#writer = SummaryWriter()

def plt_tensor_img(img,epoch,name):
    
    #plt_img = img.view(img.shape[0],3,re_size,re_size)
    grid_img = torchvision.utils.make_grid(img.cpu().data, nrow=10)
    plt.imshow(np.uint8(grid_img.permute(1, 2, 0)*255))
    plt.show()
    #grid = utils.make_grid(images)
    writer.add_image(name, grid_img, global_step=epoch)

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
    output = model(x)
    ans=torch.max(output,1)[1].squeeze()
    ans_softmax = F.softmax(output,1).squeeze()
    #loss = mse_loss(xhat, x)
    #print(loss.item())
    return ans_softmax,ans.item()


if __name__ == '__main__':
    
    print(torch.cuda.get_device_properties(0))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    re_size=256
    batch_size = 60
    
    AE = ConvNetwork_classification().to(device)
    AE.load_state_dict(torch.load("./save_pth_class/339_class_0.03645724384114146_99.40191650390625_98.64865112304688.pkl"))
    
    val_path=r"D:\Lab702\AOI\Train_Foot_position\class_val"

    val_data =  datasets.ImageFolder(val_path,transform=transforms.Compose([
        transforms.Resize(size=(re_size,re_size)),#(h,w)
        transforms.ToTensor(),
    ]))
    
    val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=batch_size,shuffle=True,num_workers=0)
    
    correct_val = 0
    with torch.no_grad():
            AE.eval()
            dis_img=None
            for step, (img,label) in enumerate(val_loader):
                input_tensor = Variable(img).to(device)
                labels = Variable(label).to(device)
                output = AE(input_tensor)
                #reconst_loss = loss_fun(output,labels)
                #val_loss_reg +=reconst_loss.item()
                ans=torch.max(output,1)[1].squeeze()
                correct_val += (ans.cpu() == labels.cpu()).float().sum()
            val_accuracy = 100 * correct_val / float(len(val_data))
            
    print(f'val_caccurary:{val_accuracy}%')
            
        
#%%
    nf_count,af_count,n_count,an_count = 0,0,0,0
    status=['normal','abnormal']
    normal_path_file =[ r"D:\Lab702\AOI\Train_Foot_position\class_sampling",
                       "D:\\Lab702\\AOI\\Train_Foot_position\\class_val\\0_normal"]
    with torch.no_grad():
        AE.eval()
        n_count=0
        for normal_path in normal_path_file:
            for img_name in os.listdir(normal_path):
                #print(img_name)
                img_path="{}/{}".format(normal_path,img_name)
                ans_softmax,ans = detect_aoi(img_path,AE)
                nf_count+=1
                #print(status[ans])
                if not(ans):
                    n_count+=1
                else:
                    print(img_path)
                    print(status[ans])
                    img = Image.open(img_path)
                    plt.imshow(img)
                    plt.show()
#%%
    abnormal_path = r"D:\Lab702\AOI\Train_Foot_position\class_val\1_abnormal"
    status=['normal','abnormal']
    with torch.no_grad():
        AE.eval()
        for img_name in os.listdir(abnormal_path):
            #print(img_name)
            img_path="{}/{}".format(abnormal_path,img_name)
            ans_softmax,ans = detect_aoi(img_path,AE)
            af_count+=1
            #print(status[ans])
            if (ans):
                an_count+=1
            else:
                print(img_path)
                print(status[ans])
                img = Image.open(img_path)
                plt.imshow(img)
                plt.show()
#%%
    print("normal:{}/{}".format(n_count,nf_count))
            
    print("abnormal:{}/{}".format(an_count,af_count))
    
    print("正確率:{:3.1f}%".format(100*(n_count+an_count)/(af_count+nf_count)))
#%%
    # path_file=[r'D:\Lab702\AOI\Train_Foot_position\class_train\0_normal',
    #            r'D:\Lab702\AOI\Train_Foot_position\class_train\1_abnormal',
    #            r'D:\Lab702\AOI\Train_Foot_position\class_val\0_normal',
    #            r'D:\Lab702\AOI\Train_Foot_position\class_val\1_abnormal',
    #            r'D:\Lab702\AOI\Train_Foot_position\class_sampling',]
    # with torch.no_grad():
    #     AE.eval()
    #     n_count=0
    #     path = 'output_data_0.txt'
    #     with open(path, 'w') as f:
    #         for img_path in path_file:
    #             for img_name in os.listdir(img_path):
    #                 #print(img_name)
    #                 input_img_path="{}/{}".format(img_path,img_name)
    #                 ans_softmax,ans = detect_aoi(input_img_path,AE)
    #                 write = f'{img_name} {ans_softmax[0].item()} {ans_softmax[1].item()} {ans}'
    #                 print(write)
    #                 f.write(f'{write}\n')
#%%
    