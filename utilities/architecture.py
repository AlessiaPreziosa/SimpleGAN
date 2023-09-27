import torch.nn as nn
from matplotlib import pyplot as plt
import torch
import os

from GAN.discriminator import Discriminator
from GAN.generator import Generator


# weights initialization
def init_weights(m):
    classname = m.__class__.__name__
    # Weights from Convolutional blocks are normalized to 0 mean and 0.02 standard deviation
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # γ from BatchNormalization blocks is normalized to 0 mean and 0.02 standard deviation
    # β from BatchNormalization blocks is transformed to a constant 0 vector
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot(x, y, title):
    plt.plot(x, label='Train')
    plt.plot(y, label='Validation')
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()


def initialize(hyperparams):

    # define generator and discriminator of the net
    generator = Generator()
    discriminator = Discriminator()

    # initialize weights for generator and discriminator
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    # functions for optimizers of Discriminator and Generator
    optimizerG = torch.optim.Adam(generator.parameters(), lr=hyperparams['lr'], betas=(hyperparams['beta1'], 0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=hyperparams['lr'], betas=(hyperparams['beta1'], 0.999))

    return generator, discriminator, optimizerG, optimizerD


def save_and_plot(trials, input_, target_, input_val, target_val, nome):
    t = trials.best_trial['result']
    gan = t['model']

    # fine-tune with the best hyperparameters
    gtl, gvl, dtl, dvl, fid = gan.fit(100, input_, target_, input_val, target_val)

    print(f'Saving model of {nome}...')

    checkpoint = {'Generator': gan.generator,
                  'Discriminator': gan.discriminator,
                  'OptimizerG': gan.optimizerG,
                  'OptimizerD': gan.optimizerD,
                  'Hyperparameters': t['hyper'],
                  'gtl': gtl,
                  'gvl': gvl,
                  'dtl': dtl,
                  'dvl': dvl,
                  'fid': fid,
                  }

    # models directory need to be created
    torch.save(checkpoint, f'{os.getcwd()}/models/checkpoint_{nome}.pth')

    # plotting
    plot(gtl, gvl, f'Generator: {nome}')
    plot(dtl, dvl, f'Discriminator: {nome}')

    plt.plot(fid)
    plt.title(f'FID score: {nome}')
    plt.grid()
    plt.show()

    return gan

