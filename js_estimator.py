import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import optim, nn
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from tqdm import tqdm


class JSDEstimator:
    def __init__(self, network, iterations=1, device="cuda") -> None:
        super().__init__()
        self._discriminator = network
        self._optimizer = optim.Adam(self._discriminator.parameters(), lr=5e-7)
        self._iterations = iterations
        self._device = device
        self._criterion = nn.CrossEntropyLoss()
        self._discriminator.to(self._device)

    def __call__(self, X, Y):
        X, Y = X.to(self._device), Y.to(self._device)
        mask = torch.Tensor([1, 0]).reshape((2,)).to(self._device).type(torch.long)

        for i in range(self._iterations):
            preds = self._discriminator(torch.cat((X, Y), dim=0))
            loss = self._criterion(preds, mask)
            self._optimizer.zero_grad()
            (loss * 1.0).backward()
            self._optimizer.step()
        preds = self._discriminator(torch.cat((X, Y), dim=0))
        loss = self._criterion(preds, mask)
        return -loss


network = resnet18(pretrained=True)
network.fc = nn.Linear(512, 2)

jsdestimator = JSDEstimator(network)

X = ToTensor()(Image.open("test.jpg").resize((128, 128))).unsqueeze(0)
y = torch.zeros(1, 3, 128, 128, requires_grad=True)
# y = X.clone()

optimizer = optim.Adam((y,), lr=1e-1)

# X, y = X.to("cuda"), y.to("cuda")

plt.ion()
indicator = tqdm(range(100000))
for i in indicator:
    loss = jsdestimator(X, y)

    optimizer.zero_grad()
    (loss * 1000).backward()
    # y = torch.clamp(y, 0, 1)
    optimizer.step()
    indicator.set_postfix({"loss": loss.item(), "y_grad_max": y.grad.max().item(), "y_grad_mean": y.grad.mean().item()})
    if i % 30 == 0:
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.imshow(X[0].cpu().detach().numpy().transpose(1, 2, 0))

        _y = y[0].cpu().detach().numpy().transpose(1, 2, 0)
        _y = (_y - _y.min()) / (_y.max() - _y.min())
        plt.subplot(2, 1, 2)
        plt.imshow(_y)
        plt.show()
        plt.pause(0.001)

plt.ioff()
