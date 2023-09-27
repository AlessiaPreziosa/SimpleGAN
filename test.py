# import libraries
import warnings

from torchvision.datasets import ImageFolder
from GAN.SimpleGAN import SimpleGAN
from utilities.dataset import *
from utilities.architecture import *

warnings.filterwarnings('ignore')

'''This script is exclusively for testing the model: the three checkpoints have to be saved in the models directory. 
    What kind of outfit is created by the model?'''

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MacOS software system
if device != 'cuda':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# load images from test dataset ...
test_outfit = f'{root_dir}outfits/test'
test_loader = ImageFolder(test_outfit, transform=test_transform)
# ... and organize them in 4 different DataLoaders: iter_tops, iter_bottoms, iter_shoes, iter_accessories
iterTe_tops, iterTe_bottoms, iterTe_shoes, iterTe_accessories = DataLoad(test_loader)


def test(cloth, target):
    checkpoint = torch.load(f'{os.getcwd()}/models/checkpoint_{cloth}.pth')
    gan = SimpleGAN(device, cloth,
                    checkpoint['Generator'].eval(), checkpoint['Discriminator'].eval(),
                    checkpoint['OptimizerG'], checkpoint['OptimizerD'],
                    checkpoint['Hyperparameters']['lmd'])
    param = gan.step(iterTe_tops, target, 'test')
    print(cloth.upper()[0] + cloth[1:])
    print('Generator Loss: ', param[0])
    print('Discriminator Loss: ', param[1])
    print('FID score: ', param[2])
    return gan


gan_bottoms = test('bottoms', iter(iterTe_bottoms))
gan_shoes = test('shoes', iter(iterTe_shoes))
gan_accessories = test('accessories', iter(iterTe_accessories))

indexes = torch.arange(0, 128).float().multinomial(num_samples=1)

# What's my outfit?

OutfitGAN = [gan_bottoms.generator.to('cpu').eval(), gan_shoes.generator.to('cpu').eval(),
             gan_accessories.generator.to('cpu').eval()]

print('How many outfits do you wanna check?')
counts = int(input())
j = 0

bottoms = iter(iterTe_bottoms)
shoes = iter(iterTe_shoes)
accessories = iter(iterTe_accessories)

i = indexes
for X, _ in iterTe_tops:

    b, _ = next(bottoms)
    s, _ = next(shoes)
    a, _ = next(accessories)
    imgs = [X[i].permute(0, 2, 3, 1) / 2 + 0.5]
    for generator in OutfitGAN:
        imgs.append(generator(X[i]).permute(0, 2, 3, 1) / 2 + 0.5)
    outfit = torch.cat((imgs[0], imgs[1], imgs[2], imgs[3]))
    real = torch.cat((X[i].permute(0, 2, 3, 1) / 2 + 0.5, b[i].permute(0, 2, 3, 1) / 2 + 0.5,
                      s[i].permute(0, 2, 3, 1) / 2 + 0.5, a[i].permute(0, 2, 3, 1) / 2 + 0.5))

    fig = plt.figure(figsize=(20, 20))
    k = 0
    for idx in range(1, 9, 2):
        fig.add_subplot(4, 2, idx)
        plt.imshow(real[k].detach().numpy())
        fig.add_subplot(4, 2, idx + 1)
        plt.imshow(outfit[k].detach().numpy())
        k += 1
    plt.show()

    j += 1
    if j == counts:
        break


