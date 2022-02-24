import argparse
from train import *
import torch
from torchvision.transforms import transforms
from data_process import *
from attack_method import *
from create_dir import *
from model import *
from test import Test

parser = argparse.ArgumentParser(description='hypers')
parser.add_argument('--BATCH_SIZE', type=int, default=8)
parser.add_argument('--EPSILON', type=float, default=8)
parser.add_argument('--ALPHA', type=float, default=0.8)
args = parser.parse_args()

root = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# calculated statistics from cifar_10 dataset
# mean for the three channels of cifar_10 images
cifar_10_mean = (0.491, 0.482, 0.447)
# std for the three channels of cifar_10 images
cifar_10_std = (0.202, 0.199, 0.201)

mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

adv_set = AdvDataset(root, transform=transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set, batch_size=args.BATCH_SIZE, shuffle=False)

# For debug
print(f'number of images = {adv_set.__len__()}')

model_lists = ['resnext29_16x64d_cifar10', 'resnext29_32x4d_cifar10', 'preresnet56_cifar10', 'preresnet110_cifar10',
               'preresnet164bn_cifar10', 'seresnet110_cifar10', 'sepreresnet56_cifar10', 'sepreresnet110_cifar10',
               'diaresnet56_cifar10', 'resnet1001_cifar10', 'diapreresnet56_cifar10', 'resnet1202_cifar10',
               'resnet56_cifar10', 'resnet110_cifar10', 'diapreresnet110_cifar10']

model = FusionEnsemble(model_lists, device=device)
loss_fn = nn.CrossEntropyLoss()


benign_acc, benign_loss = epoch_benign(model, adv_loader, loss_fn, device)
print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')


# train
for method in ['fgsm', 'ifgsm', 'mifgsm']:
    adv_examples, acc, loss = train(
        model, adv_loader, method, loss_fn, std, mean, device)
    print(f'method = {method}, acc = {acc:.5f}, loss = {loss:.5f}')
    create_dir(root, method, adv_examples, adv_names)


# test
eval_model = ['resnext29_16x64d_cifar10',
              'sepreresnet56_cifar10', 'resnet1202_cifar10']
classes = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for file in ['fgsm', 'ifgsm', 'mifgsm']:
    print(f'attck with {file}')
    tester = Test(eval_model, classes, transform, device, file)
    tester.test()
