import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from AE_model import ConvAutoencoder
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter       

writer = SummaryWriter()

def plt_tensor_img(img,epoch,name):
    
    #plt_img = img.view(img.shape[0],3,re_size,re_size)
    grid_img = torchvision.utils.make_grid(img.cpu().data, nrow=10)
    plt.imshow(np.uint8(grid_img.permute(1, 2, 0)*255))
    plt.show()
    #grid = utils.make_grid(images)
    writer.add_image(name, grid_img, global_step=epoch)




if __name__ == '__main__':
    
    print(torch.cuda.get_device_properties(0))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    re_size=256
    z_dim = 64
    batch_size = 60
    num_epochs = 2500
    learning_rate = 1.2e-06
    
    save_file="save_pth"
    if not os.path.isdir(save_file):
            os.mkdir(save_file)
            
    
    AE = ConvAutoencoder(z_dim).to(device)
    AE.load_state_dict(torch.load("./save_pth/AE_0.00015204532610368915.pkl"))
    
    loss_fun =  nn.MSELoss()
    
    optimizer_AE = optim.RMSprop(AE.parameters(), lr=learning_rate)
    scheduler_AE = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AE, 'min',factor=0.1, patience=500, verbose=True)
    
    train_path=r"D:\Lab702\AOI\Train_Foot_position\train"
    val_path=r"D:\Lab702\AOI\Train_Foot_position\val"
    #noise_path=r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\2_abnormal"
    
    train_data = datasets.ImageFolder(train_path,transform=transforms.Compose([
        transforms.Resize(size=(re_size,re_size)),#(h,w)
        transforms.RandomHorizontalFlip(p=0.5),#依據p概率水平翻轉
        transforms.RandomVerticalFlip(p=0.5),#依據p概率垂直翻轉
        transforms.ToTensor(),
    ]))
    val_data =  datasets.ImageFolder(val_path,transform=transforms.Compose([
        transforms.Resize(size=(re_size,re_size)),#(h,w)
        transforms.ToTensor(),
    ]))

    print(train_data.classes)#获取标签
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=batch_size,shuffle=True,num_workers=0)
    
    low_avg_loss = 100
    
    for epoch in range(num_epochs):
        AE.train()
        i = 0
        train_loss_reg,val_loss_reg = 0,0
        dis_img=None
        for step, (img,_) in enumerate(train_loader):
            
            input_tensor = Variable(img).to(device)
            
            AE.zero_grad()
            output = AE(input_tensor)
            
            reconst_loss = loss_fun(output,input_tensor)
            reconst_loss.backward()
            
            optimizer_AE.step()
    
            train_loss_reg +=reconst_loss.item()
            
            #print('iterate [{}/{}], loss: {:.4f}'.format(step+1,len(train_loader),reconst_loss))
            if(step==len(train_loader)-2):
                dis_img = output
        
        print("="*25)
        avg_loss = train_loss_reg / len(train_loader)
        print("epoch [{}/{}],avg_loss:{},lr:{}".format(epoch+1,num_epochs,avg_loss,optimizer_AE.param_groups[0]['lr']))
        writer.add_scalar('Train/avg_loss\\', avg_loss, epoch)
        writer.flush()
        #plt_tensor_img(dis_img,epoch,'Train/output\\')
        
        with torch.no_grad():
            AE.eval()
            dis_img=None
            for step, (img,_) in enumerate(val_loader):
                input_tensor = Variable(img).to(device)
                output = AE(input_tensor)
                reconst_loss = loss_fun(output,input_tensor)
                val_loss_reg +=reconst_loss.item()
                if(step==len(val_loader)-2):
                    dis_img = output
            avg_val_loss = val_loss_reg / len(val_loader)
            print("epoch [{}/{}],avg_val_loss:{}".format(epoch+1,num_epochs,avg_val_loss))
            
            writer.add_scalar('Train/avg_val_loss\\', avg_val_loss, epoch)
            writer.flush()
            #plt_tensor_img(dis_img,epoch,'Train/val_output\\')
            
        scheduler_AE.step(avg_val_loss)
        torch.cuda.empty_cache()
        
        if(low_avg_loss>avg_val_loss and avg_val_loss<=0.0001525):
            low_avg_loss = avg_val_loss
            torch.save(AE.state_dict(), '{}/{}_AE_{}.pkl'.format(save_file,epoch,avg_val_loss))
        
            