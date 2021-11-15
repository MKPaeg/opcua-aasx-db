import torch.nn as nn

def mobile_block(in_dim, out_dim, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=stride, padding=1, groups=in_dim),
        nn.BatchNorm2d(in_dim),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
        # nn.ReLU(inplace=True),
        nn.ReLU6(inplace=True),
    )

class MobileNet(nn.Module):
    def __init__(self, channeldepth = int(256 / 32), num_classes=6):
        super(MobileNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32*channeldepth, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32*channeldepth),
            nn.ReLU(inplace=True),

            mobile_block(32*channeldepth, 64*channeldepth),
            mobile_block(64*channeldepth, 128*channeldepth, 2),
            mobile_block(128*channeldepth, 128*channeldepth),
            mobile_block(128*channeldepth, 256*channeldepth, 2),
            mobile_block(256*channeldepth, 256*channeldepth),
            mobile_block(256*channeldepth, 512*channeldepth, 2),
            #*[mobile_block(512*channeldepth, 512*channeldepth) for _ in range(5)],
            *[mobile_block(512 * channeldepth, 512 * channeldepth) for _ in range(3)],
            mobile_block(512*channeldepth, 1024*channeldepth, 2),
            mobile_block(1024*channeldepth, 1024*channeldepth),

            nn.AvgPool2d(7),
        )
        self.classifier = nn.Linear(1024*channeldepth, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x