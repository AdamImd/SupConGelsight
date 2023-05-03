import torch.nn as nn
import torchvision
import torch

## create a base network of efficentnet_v2_m and change the last layer to 6 classes
class Net(nn.Module):
    def __init__(self, weights=None, path=None):
        super(Net, self).__init__()
        self.base_model = torchvision.models.efficientnet_v2_m(weights=weights)
        self.input_res = (224, 224)

        # change the last layer to 6 classes
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1000, 6)
        
        if path is not None:
            self.load(path)

    def forward(self, x):

        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

    def infrence(self, x):
        torch.eval()
        x = self.forward(x)
        return x