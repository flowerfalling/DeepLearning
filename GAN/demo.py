import matplotlib.pyplot as plt
from torchvision import transforms, datasets

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.CelebA(root=r'..\data', transform=transform)
plt.imshow(trainset[0][0].numpy().transpose((1, 2, 0)))
plt.show()
pass
