import torch.nn as nn
import torch.utils.data

from utils.mobilenet import MobileNet
import torch.optim as optim
import utils.data_load_v2 as data_load
import glob
import os
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
log_dir = "./log_dir"
summary  = SummaryWriter(log_dir)

def train(model, criterion, train_dataset, train_dataloader, optimizer, scheduler, train_path, batch_size=8, epochs=30, path="./model_v2"):
    if torch.cuda.is_available():
        model = MobileNet(num_classes=6).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_dataset = data_load.Sup_Img_Dataset(file, data_load.ImageTransform())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if not os.path.exists(path):
        os.mkdir(path)

    for epoch in range(epochs):
        train_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader, desc=f"epoch {epoch}")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            train_loss += loss.item()
            epoch_loss += loss.item()
            if i % 5 == 4:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, train_loss / 5))
                train_loss = 0.0


        scheduler.step()
        summary.add_scalar('loss/epoch_loss', epoch_loss/batch_size, epoch)
        saved_path = os.path.join(path, f"tomato_ripeness_{epoch}_" + ".pt")
        # torch.save(model.state_dict(), saved_path)
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer" :optimizer.state_dict(),
                    "epoch": epoch/batch_size,
                    "loss" : epoch_loss
                    }, saved_path)

    print('Finished Training')

if __name__ =="__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))

    batch_size = 8
    epochs = 55

    model = None
    if torch.cuda.is_available():
        model = MobileNet(num_classes=6).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.9)
    # lr scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)

    train_path = "F:\_data\_Food\pytorch_tomato_v2"
    file = glob.glob(train_path + "\*.raw")

    train_dataset = data_load.Sup_Img_Dataset(file, data_load.ImageTransform())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train(model, criterion, train_dataset, train_dataloader, optimizer, scheduler, train_path, batch_size=batch_size, epochs=epochs,
          path="./model_v3")

