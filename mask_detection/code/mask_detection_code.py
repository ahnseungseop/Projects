import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob
import warnings
from tqdm.notebook import tqdm
import numpy as np

#%%

class MNIST(Dataset):

    def __init__(self, data_dir, folder, transform=None):
        
        self.img_list = sorted(glob(os.path.join(data_dir, folder, '*')))
               
        self.transform = transform    
        
                

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, idx):

        # write your codes here
        
        data_path = self.img_list[idx]
        img = io.imread(data_path)
        
        if self.transform:
            img = self.transform(img) 
            
        img_name = os.path.basename(data_path)
        for i in range(0,10) :
            if i == int(img_name[-5]): #-5,6
                label = torch.tensor(i)
                break
            else :
                continue
            
        
        return  img, label
    
#%%

d =  "./data"

transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#%%

BATCH_SIZE = 64

train_set = MNIST(data_dir=d, folder='train', transform = transform_train)
test_set = MNIST(data_dir=d, folder='test', transform = transform_test)    


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#%%

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

figure = plt.figure()
num_of_images = 1
for index in range(1, num_of_images + 1):
    plt.subplot(1, 1, index)
    plt.axis('off')
    plt.imshow(images[index][index].numpy().squeeze())
    print(images[index][index].numpy().squeeze().shape)
    
#%%

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

#%%

def conv1x1(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#%%

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, quantize=False):
        super(BasicBlock, self).__init__()
        self.quantize = quantize
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
               
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        self.skip_add = nn.quantized.FloatFunctional()
        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Notice the addition operation in both scenarios
        if self.quantize:
            out = self.skip_add.add(out, identity)
        else:
            out += identity

        out = self.relu(out)

        return out
    
#%%

class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, mnist=False, quantize=False):
        super(ResNet, self).__init__()
        self.quantize = quantize
        if mnist:
            num_channels = 3
        else:
            num_channels = 3
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, quantize=self.quantize))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, quantize=self.quantize))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # Input are quantized
        if self.quantize:
            x = self.quant(x)
    
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        
        if self.quantize:
            x = self.dequant(x)
        
        return x

    def forward(self, x):
         
        return self._forward_impl(x)
    
#%%
    
def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    trn_loss= 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = F.nll_loss(F.log_softmax(output, dim=-1), target)
        trn_loss +=F.nll_loss(F.log_softmax(output, dim=-1), target, reduction='sum').item()
        pred_tr = output.argmax(dim=1,keepdim=True)
        correct += pred_tr.eq(target.view_as(pred_tr)).sum().item()
        
        loss.backward()
        
        optimizer.step()
           
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}' .format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    trn_loss /= len(train_loader.dataset)
    tr_acc = 100*correct / len(train_loader.dataset)
    print('train_loss: {:.4f}, tr_acc: {:.2f}%'.format(trn_loss, tr_acc))
    
    return trn_loss, tr_acc

#%%

def test(model, device, test_loader):
    model.to(device)
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=-1), target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100*correct / len(test_loader.dataset)
    
    return test_loss, acc
   
    
#%%

def main():
     
    epochs = 40
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = True 
    no_cuda = False
    
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = ResNet(num_classes=3, mnist=True).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.001)
    
    args = {}
    
    args["log_interval"] = log_interval
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(args, model, device, train_loader, optimizer, epoch)
        tst_loss, acc =  test(model, device, test_loader)
        
        print('[{}] Test Loss : {:.4f}, Test Acc: {:.2f}%'.format(epoch, tst_loss, acc))
      
    if (save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

main()

#%%

device = 'cpu'
encoder = ResNet(num_classes=3, mnist=True)
loaded_dict_enc = torch.load('mnist_cnn.pt', map_location=device)
encoder.load_state_dict(loaded_dict_enc)
test(model=encoder, device=device, test_loader=test_loader)

#%%
with torch.no_grad() :
    
    index = 202
        
    item = train_set[index]
        
    image = item[0]
        
    image2 = image.unsqueeze(0)
        
    true_target = item[1]
        
    model = ResNet(num_classes=3, mnist=False).to(device)
        
    model.load_state_dict(torch.load('mnist_cnn.pt'),strict=False)
        
    model.eval()
        
    prediction = model(image2)
        
    prediction_class = np.argmax(prediction)
        
    image = image.reshape(1, 3, 300, 300)
        
    plt.imshow(image[0][0].numpy().squeeze())
        
    plt.title(f'Prediction: {prediction_class} - Actual target: {true_target}')
        
    plt.show()
    

#%%

def prediction(x,dataset) :

    with torch.no_grad() :

        index = x

        item = dataset[index]

        image = item[0]

        image2 = image.unsqueeze(0)
        
        model = ResNet(num_classes=3, mnist=False).to(device)

        model.load_state_dict(torch.load('mnist_cnn.pt'),strict=False)

        model.eval()

        prediction = model(image2)

        prediction_class = np.argmax(prediction)
    
    return prediction_class   

#%%

p_class=[]
for i in tqdm(range(0,1000)) :
    a = prediction(i, MNIST(data_dir=d, folder='test', transform = transform_test))
    a=int(a.numpy())
    p_class.append(a)
    
print(p_class)

#%%

r_class=[]
dataset = MNIST(data_dir=d, folder='test', transform = transform_test)      
for j in tqdm(range(0,1000)) :
    a = dataset[j][1].numpy()
    a=int(a)
    r_class.append(a)

print(r_class)

#%%

from sklearn.metrics import confusion_matrix

# 열 예측값
# 행 실제값

confusion_matrix(r_class, p_class)

