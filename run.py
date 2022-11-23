import os
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, datasets, models
import torchvision
import pylab
import matplotlib.pyplot as plt
from Net import *

def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def image_show(images):# image_show(grid_img)
    images = images.numpy()
    images = images.transpose((1, 2, 0))
    print(images.shape)
    plt.imshow(images)
    plt.show()



save_file="11_save"
if not os.path.isdir(save_file):
        os.mkdir(save_file)

if __name__ == '__main__':
    
    print(torch.cuda.get_device_properties(0))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    re_size=224
    z_dim = 64
    batch_size = 50
    num_epochs = 5000
    learning_rate = 1.0e-3
    n = 6 #number of test sample
    
    model = Autoencoder(z_dim,re_size).to(device)
    #ConvAutoencoder().to(device)
    
    model.load_state_dict(torch.load('224_save/best_2840_0.0001145708083640784.pkl'))
    
    mse_loss = nn.MSELoss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.33, patience=250, verbose=True)
    
    
    train_path=r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\train"
    #dev_path="./datasets/unet_crop_Dev"
    
    train_data = datasets.ImageFolder(train_path,transform=transforms.Compose([
        transforms.Resize(size=(re_size,re_size)),#(h,w)
        #transforms.RandomCrop(size=(256,256), padding=5),
        #transforms.ColorJitter(brightness=0.2, contrast=0.5,saturation=0.5),
        transforms.RandomHorizontalFlip(p=0.5),#依據p概率水平翻轉
        transforms.RandomVerticalFlip(p=0.5),#依據p概率垂直翻轉
        #transforms.RandomRotation((-45,45)),#隨機角度旋轉
        #transforms.RandomGrayscale(p=0.4),
        transforms.ToTensor()
    ]))
    
    print(train_data.classes)#获取标签

    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0)
    
    losses = np.zeros(num_epochs)
    
    best_low_loss=100
    
    for epoch in range(num_epochs):
        model.train()
        i = 0
        train_loss_reg = 0
        for img,_ in train_loader:

            x = img.view(img.size(0), -1)
            x = Variable(x).to(device)
            #x = Variable(img).to(device)
            
            xhat = model(x).to(device)
            
            # plt_xhat = xhat.view(x.shape[0],3,re_size,re_size)
            # grid_img = torchvision.utils.make_grid(plt_xhat.cpu().data, nrow=10)
            # plt.imshow(np.uint8(grid_img.permute(1, 2, 0)*255))
            # plt.show()
            
            # 出力画像（再構成画像）と入力画像の間でlossを計算
            loss = mse_loss(xhat, x)
            losses[epoch] = losses[epoch] * (i / (i + 1.)) + loss * (1. / (i + 1.))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            train_loss_reg +=loss.item()
            
            print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1,num_epochs,loss))
        
        plt_xhat = xhat.view(x.shape[0],3,re_size,re_size)
        grid_img = torchvision.utils.make_grid(plt_xhat.cpu().data, nrow=10)
        plt.imshow(np.uint8(grid_img.permute(1, 2, 0)*255))
        plt.show()
        
        avg_train_loss = train_loss_reg/float(len(train_loader))
        if(avg_train_loss<=best_low_loss and avg_train_loss<=0.0002):
            torch.save(model.state_dict(), '{}/best_{}_{}.pkl'.format(save_file,epoch+1,losses[epoch]))
            best_low_loss=avg_train_loss
        print("*"*10,losses[epoch],avg_train_loss)
        print("*"*10,optimizer.param_groups[0]['lr'])
        scheduler.step(losses[epoch])
        
        torch.cuda.empty_cache()
        
    plt.figure()
    pylab.xlim(0, num_epochs)
    plt.plot(range(0, num_epochs), losses, label='loss')
    plt.legend()
    #plt.savefig(os.path.join("./save/", 'loss.pdf'))
    plt.close()


# test_dataset = MNIST('./data', train=False,download=True, transform=img_transform)
# test_1_9 = Mnisttox(test_dataset,[1,9])
# test_loader = DataLoader(test_1_9, batch_size=len(test_dataset), shuffle=True)

# for img,_ in test_loader:
#     x = img.view(img.size(0), -1)

#     if cuda:
#         x = Variable(x).cuda()
#     else:
#         x = Variable(x)

#     xhat = model(x)
#     x = x.cpu().detach().numpy()
#     xhat = xhat.cpu().detach().numpy()
#     x = x/2 + 0.5
#     xhat = xhat/2 + 0.5

# # サンプル画像表示
# plt.figure(figsize=(12, 6))
# for i in range(n):
#     # テスト画像を表示
#     ax = plt.subplot(3, n, i + 1)
#     plt.imshow(x[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # 出力画像を表示
#     ax = plt.subplot(3, n, i + 1 + n)
#     plt.imshow(xhat[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # 入出力の差分画像を計算
#     diff_img = np.abs(x[i] - xhat[i])

#     # 入出力の差分数値を計算
#     diff = np.sum(diff_img)

#     # 差分画像と差分数値の表示
#     ax = plt.subplot(3, n, i + 1 + n * 2)
#     plt.imshow(diff_img.reshape(28, 28),cmap="jet")
#     #plt.gray()
#     ax.get_xaxis().set_visible(True)
#     ax.get_yaxis().set_visible(True)
#     ax.set_xlabel('score = ' + str(diff))

# plt.savefig("./save/result.png")
# plt.show()
# plt.close()
    