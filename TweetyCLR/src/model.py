import torch.nn as nn
import torch.nn.functional as F
import torch

# class Encoder(nn.Module):    
#     def __init__(self):
#         super().__init__()
#         # TweetyNet Front End but with pooling in both time and frequency dimension. 
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
#         self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
#         self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
#         self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

#         # Let's add a few linear layers to reduce the dimensionality down further
#         self.fc1 = nn.Linear(3456, 1000)
#         self.fc2 = nn.Linear(1000, 256)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool3(x)
#         x = F.relu(self.conv4(x))
#         x = self.pool4(x)

#         x = x.view(x.size(0), -1)

#         x = self.fc1(x)
#         x = self.fc2(x)

#         return x
    
    # def get_batch_stats(self, x):
    #     batch_mean = torch.mean(x, dim=[0, 2, 3])
    #     batch_var = torch.var(x, dim=[0, 2, 3])
    #     return batch_mean, batch_var

    
    # def get_running_stats(self):
    #     running_means = []
    #     running_vars = []
    #     for name, module in self.named_modules():
    #         if isinstance(module, nn.BatchNorm2d):
    #             running_means.append((name, module.running_mean.cpu().detach().numpy()))
    #             running_vars.append((name, module.running_var.cpu().detach().numpy()))

    #     return running_means, running_vars

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3,1,padding=1) 
        self.conv2 = nn.Conv2d(8, 8, 3,2,padding=1) 
        self.conv3 = nn.Conv2d(8, 16,3,1,padding=1) 
        self.conv4 = nn.Conv2d(16,16,3,2,padding=1) 
        self.conv5 = nn.Conv2d(16,24,3,1,padding=1) 
        self.conv6 = nn.Conv2d(24,24,3,2,padding=1) 
        self.conv7 = nn.Conv2d(24,32,3,1,padding=1) 
        self.conv8 = nn.Conv2d(32,24,3,2,padding=1)
        self.conv9 = nn.Conv2d(24,24,3,1,padding=1)
        self.conv10 = nn.Conv2d(24,16,3,2,padding=1)
        
        self.bn1 = nn.BatchNorm2d(8) 
        self.bn2 = nn.BatchNorm2d(8) 
        self.bn3 = nn.BatchNorm2d(16) 
        self.bn4 = nn.BatchNorm2d(16) 
        self.bn5 = nn.BatchNorm2d(24) 
        self.bn6 = nn.BatchNorm2d(24) 
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(24)
        self.bn9 = nn.BatchNorm2d(24)
        self.bn10 = nn.BatchNorm2d(16)


        self.relu = nn.ReLU()       
        self.dropout = nn.Dropout2d(
        )
        self._to_linear = 320
        
    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x))) 
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x))) 
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))

        # x = F.relu((self.conv1(x)))
        # x = F.relu((self.conv2(x))) 
        # x = F.relu((self.conv3(x)))
        # x = F.relu((self.conv4(x))) 
        # x = F.relu((self.conv5(x)))
        # x = F.relu((self.conv6(x))) 
        # x = F.relu((self.conv7(x)))
        # x = F.relu((self.conv8(x)))
        # x = F.relu((self.conv9(x)))
        # x = F.relu((self.conv10(x)))


        x = x.view(-1, 320)

        
        return x
    
#     def get_batch_stats(self, x):
#         batch_mean = torch.mean(x, dim=[0, 2, 3])
#         batch_var = torch.var(x, dim=[0, 2, 3])
#         return batch_mean, batch_var

    
#     def get_running_stats(self):
#         running_means = []
#         running_vars = []
#         for name, module in self.named_modules():
#             if isinstance(module, nn.BatchNorm2d):
#                 running_means.append((name, module.running_mean.cpu().detach().numpy()))
#                 running_vars.append((name, module.running_var.cpu().detach().numpy()))
#         return running_means, running_vars