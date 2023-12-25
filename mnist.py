from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
mnist_train = MNIST(root="/MNIST_data", train=True, download=True, transform=None)
print(len(mnist_train))
print(mnist_train[0][0])
image = mnist_train[0][0]
plt.imshow(image)
plt.show()
print(mnist_train[0][1])
