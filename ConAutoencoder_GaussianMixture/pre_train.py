import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from model import ConvAutoencoder
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


def set_center(model, dataloader, eps=0.1):
    """Initializing the center for the hypersphere"""
    model.eval()
    z_ = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.float().to(device)
            z = model.encode(x)
            z_.append(z.detach())
    z_ = torch.cat(z_)
    c = torch.mean(z_, dim=0)
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c

def save_weights(model, dataloader,save_file,avg_loss):
    """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
    c = self.set_c(model, dataloader)
    net = network(self.args.latent_dim).to(self.device)
    state_dict = model.state_dict()
    net.load_state_dict(state_dict, strict=False)
    torch.save({'center': c.cpu().data.numpy().tolist(),
                'net_dict': net.state_dict()}, '{}/Deep_SVDD_{}.pth'.format(save_file,avg_loss))

if __name__ == '__main__':
    
    print(torch.cuda.get_device_properties(0))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    re_size=64
    z_dim = 32
    batch_size = 60
    num_epochs = 500
    learning_rate = 1.0e-3
    
    save_file="save_pth"
    if not os.path.isdir(save_file):
            os.mkdir(save_file)
            
    
    AE = ConvAutoencoder(z_dim).to(device)
    
    loss_fun =  nn.MSELoss()
    
    optimizer_AE = optim.Adam(AE.parameters(), lr=learning_rate)
    scheduler_AE = optim.lr_scheduler.ReduceLROnPlateau(optimizer_AE, 'min',factor=0.33, patience=100, verbose=True)
    
    train_path=r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\train"
    #noise_path=r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\2_abnormal"
    
    train_data = datasets.ImageFolder(train_path,transform=transforms.Compose([
        transforms.Resize(size=(re_size,re_size)),#(h,w)
        transforms.RandomHorizontalFlip(p=0.5),#依據p概率水平翻轉
        transforms.RandomVerticalFlip(p=0.5),#依據p概率垂直翻轉
        transforms.ToTensor(),
    ]))

    print(train_data.classes)#获取标签
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0)
    
    low_avg_loss = 100
    
    for epoch in range(num_epochs):
        AE.train()
        i = 0
        train_loss_reg = 0
        dis_img=None
        for step, (img,_) in enumerate(train_loader):
            
            input_tensor = Variable(img).to(device)
            
            AE.zero_grad()
            output = AE(input_tensor)
            
            reconst_loss = loss_fun(output,input_tensor)
            reconst_loss.backward()
            
            optimizer_AE.step()
    
            train_loss_reg +=reconst_loss.item()
            
            print('iterate [{}/{}], loss: {:.4f}'.format(step+1,len(train_loader),reconst_loss))
            if(step==len(train_loader)-2):
                dis_img = output
                


    
        avg_loss = train_loss_reg / len(train_loader)
        print("epoch [{}/{}],avg_loss:{},lr:{}".format(epoch+1,num_epochs,avg_loss,optimizer_AE.param_groups[0]['lr']))
        writer.add_scalar('Train/avg_loss\\', avg_loss, epoch)
        writer.flush()
        plt_tensor_img(dis_img,epoch,'Train/output\\')
        
        scheduler_AE.step(avg_loss)
        torch.cuda.empty_cache()
        
        if(low_avg_loss>avg_loss):
            low_avg_loss = avg_loss
            torch.save(AE.state_dict(), '{}/{}_AE_{}.pkl'.format(save_file,epoch,avg_loss))
            