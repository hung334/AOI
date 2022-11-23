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
from PIL import Image

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
    dist = model(x).to(device)
    
    #loss = mse_loss(xhat, x)
    #print(loss.item())
    return dist #loss.item()
    #label_pred = outputs.max(1)[1].squeeze().cpu().data.numpy()
    #plt.imshow(img)

def eval(net, c, dataloader, device):
    """Testing the Deep SVDD model"""

    scores = []
    labels = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=1)

            scores.append(score.detach().cpu())
            labels.append(y.cpu())
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
    return labels, scores





if __name__ == '__main__':
    
    print(torch.cuda.get_device_properties(0))
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    re_size=64
    z_dim = 32
    batch_size = 1
    
    model = ConvNetwork(z_dim).to(device)#Autoencoder(z_dim,re_size).to(device)
    model.load_state_dict(torch.load('save_pth/9682_net_0.0001466572767822072.pkl'))
    model.eval()
    
    center = torch.load("center.pt")
    # center=torch.tensor([-15.8139,  -7.3157,  -1.6011,  -8.8937,  11.5278,  10.7086,   4.1020,
    #       4.7148, -13.7001,  -7.5465,  -7.0363,  -6.4506, -14.1850,   8.4833,
    #      11.3652,  14.4726,  -8.4314,   6.1765,  11.4109,  -9.4586, -14.4696,
    #      14.9807,  -0.8524,   4.8178,  -6.1242,  -9.7496,  -5.9566,  -3.7315,
    #      16.2308,  14.0561,   7.3507,   3.1553]).to(device)
    
    img_path="./test/ok_1.png"
    dist = detect_aoi(img_path,model)
    
    score = torch.sum((dist - center) ** 2, dim=1)
    print(score.item())
    
    
    normal_total=[]
    abnormal_total=[]
    
    normal_path = r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\0_normal\123"
    
    for img_name in os.listdir(normal_path):
        #print(i)
        img_path="{}/{}".format(normal_path,img_name)
        dist = detect_aoi(img_path,model)
        normal_total.append(torch.sum((dist - center) ** 2, dim=1))
        #total.append(output)
    
    abnormal_path = r"D:\Lab702\AOI\ganomaly-master\data\AOI_part2\test_val\1_abnormal\123"
    
    for img_name in os.listdir(abnormal_path):
        #print(i)
        img_path="{}/{}".format(abnormal_path,img_name)
        dist = detect_aoi(img_path,model)
        abnormal_total.append(torch.sum((dist - center) ** 2, dim=1))
        #total.append(output)
        if(torch.sum((dist - center) ** 2, dim=1)<=0.0007):
            img = Image.open(img_path)
            plt.imshow(img)
            plt.show()
            
    normal_total = np.array(normal_total)
    abnormal_total = np.array(abnormal_total)
    
    pass