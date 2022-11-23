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

def set_center(model, dataloader, eps=0.1):
    model.eval()
    z_ = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.float().to(device)
            z = model.encode(x)
            z_.append(z.detach())
    z_ = torch.cat(z_)#list to tensor
    c = torch.mean(z_, dim=0)
    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


if __name__ == '__main__':
    
    
    print(torch.cuda.get_device_properties(0))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    save_file="save_pth"
    if not os.path.isdir(save_file):
            os.mkdir(save_file)
    
    re_size=64
    z_dim = 32
    batch_size = 60
    num_epochs = 10000
    learning_rate = 1.0e-3
    
    
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
    
    center = set_center(AE,train_loader)
    torch.save(center,"center.pt")
    
    net = ConvNetwork().to(device)
    #net.load_state_dict(torch.load("./save_pth/2203_net_0.0013745987950824201.pkl"))
    state_dict = AE.state_dict()
    net.load_state_dict(state_dict, strict=False)
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=500, verbose=True)
    
    low_avg_loss = 100
    
    for epoch in range(num_epochs):
        net.train()
        i = 0
        train_loss_reg = 0
        dis_img=None
        for step, (img,_) in enumerate(train_loader):
            
            input_tensor = Variable(img).to(device)
            
            net.zero_grad()
            output = net(input_tensor)
            
            dist_loss = torch.mean(torch.sum((output - center) ** 2, dim=1))
            dist_loss.backward()
            
            optimizer.step()

            train_loss_reg +=dist_loss.item()
            
            print('iterate [{}/{}], dist_loss: {:.4f}'.format(step+1,len(train_loader),dist_loss))
            if(step==len(train_loader)-2):
                dis_img = output
                

        avg_dist_loss = train_loss_reg / len(train_loader)
        print("epoch [{}/{}],avg_dist_loss:{},lr:{}".format(epoch+1,num_epochs,avg_dist_loss,optimizer.param_groups[0]['lr']))
        #writer.add_scalar('Train/avg_loss\\', avg_loss, epoch)
        #writer.flush()
        #plt_tensor_img(dis_img,epoch,'Train/output\\')
        
        scheduler.step(avg_dist_loss)
        torch.cuda.empty_cache()
        
        if(low_avg_loss>avg_dist_loss and avg_dist_loss<0.0009):
            low_avg_loss = avg_dist_loss
            torch.save(net.state_dict(), '{}/{}_net_{}.pkl'.format(save_file,epoch,avg_dist_loss))
    
    pass