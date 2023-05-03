from model import Net
from dataset import mixed_dataset
from losses import SupConLoss
from losses import TwoCropTransform
from torchvision import transforms
import cv2

import torch

model = Net().cuda()

criterion = SupConLoss(temperature=0.07).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=200, scale=(0.8,0.8), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=std),
])
    
train_dataset = mixed_dataset('./data/', train=True, transform=TwoCropTransform(train_transform))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, pin_memory=True, num_workers=4)

test_dataset = mixed_dataset('./data/', train=False, transform=train_transform)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=4)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = torch.cat([data[0], data[1]], dim=0)
        if torch.cuda.is_available():
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            torch.cuda.synchronize()

        optimizer.zero_grad()

        output = model(data)

        bsz = label.shape[0]
        f1, f2 = torch.split(output, [bsz, bsz], dim=0)
        output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                loss.item() / len(data)), end='\r')
            
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader)) + ' '*15)


def train_network():
    model.load('./best.pth')
    if torch.cuda.is_available():
        model.cuda()
    
    for epoch in range(0, 100):
        train(epoch)
        torch.save(model.state_dict(), './models/model_{}.pth'.format(epoch))

def test_network():
    model.load('./best.pth')
    for (data, label) in enumerate(test_dataset):
        if torch.cuda.is_available():
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            torch.cuda.synchronize()

        piece_type = model.infrence(data)
        print("Predicted: {}", piece_type.cpu())
        print("Actual: {}", label.cpu())

        cv2.imshow('data', data)
        cv2.waitKey(1)

if __name__ == '__main__':
    #train_network()
    test_network()
