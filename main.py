# import libraries
import random
import warnings

from torchvision.datasets import ImageFolder
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from GAN.SimpleGAN import SimpleGAN
from utilities.dataset import *
from utilities.architecture import *

warnings.filterwarnings('ignore')

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MacOS software system
if device != 'cuda':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# load images from train dataset ...
train_outfit = f'{root_dir}outfits/train'
train_loader = ImageFolder(train_outfit, transform=train_transform)
# ... and organize them in 4 different DataLoaders: iter_tops, iter_bottoms, iter_shoes, iter_accessories
iterT_tops, iterT_bottoms, iterT_shoes, iterT_accessories = DataLoad(train_loader)

# load images from validation dataset ...
valid_outfit = f'{root_dir}outfits/val'
valid_loader = ImageFolder(valid_outfit, transform=test_transform)
# ... and organize them in 4 different DataLoaders: iter_tops, iter_bottoms, iter_shoes, iter_accessories
iterV_tops, iterV_bottoms, iterV_shoes, iterV_accessories = DataLoad(valid_loader)

# show some sample images
print("Let's show some sample images...")

indexes = torch.range(0, 127)
indexes = indexes.multinomial(num_samples=20)

show(iterT_tops, indexes)
show(iterT_bottoms, indexes)
show(iterT_shoes, indexes)
show(iterT_accessories, indexes)

print("Creating the GAN...")

# Hyperparameter tuning
hyper_space = {
    'lr': hp.choice('lr', [0.0002, 0.00046666666666666666, 0.0007333333333333333, 0.001]),
    'lmd': hp.choice('lmd', np.arange(50, 110, 10).tolist()),
    'beta1': hp.choice('beta1', np.arange(0.5, 1.0, 0.1).tolist())
}


# accessories in training
def objective_accessories(hyperparams):
    print(f"Tuning for accessories: {hyperparams} ")

    generatorA, discriminatorA, optimizerGA, optimizerDA = initialize(hyperparams)

    gan_accessories = SimpleGAN(device, 'accessories', generatorA, discriminatorA, optimizerGA, optimizerDA,
                                lmd=hyperparams['lmd'])
    (_, gvl, _, dvl, fid_scores) = gan_accessories.fit(200, iterT_tops, iterT_accessories, iterV_tops, iterV_accessories)
    loss = np.mean(np.array(gvl)) + np.mean(np.array(dvl)) + np.mean(np.array(fid_scores))
    return {'loss': loss, 'model': gan_accessories, 'hyper': hyperparams, 'status': STATUS_OK}


trials_accessories = Trials()
best_accessories = fmin(fn=objective_accessories, space=hyper_space, algo=tpe.suggest, max_evals=5,
                        trials=trials_accessories, verbose=True, show_progressbar=False)

gan_accessories = save_and_plot(trials_accessories, iterT_tops, iterT_accessories, iterV_tops, iterV_accessories,
                                'accessories')


# bottoms in training
def objective_bottoms(hyperparams):
    print(f"Tuning for bottoms: {hyperparams}")

    generatorB, discriminatorB, optimizerGB, optimizerDB = initialize(hyperparams)

    gan_bottoms = SimpleGAN(device, 'bottoms', generatorB, discriminatorB, optimizerGB, optimizerDB,
                            lmd=hyperparams['lmd'])
    (_, gvl, _, dvl, fid_scores) = gan_bottoms.fit(200, iterT_tops, iterT_bottoms, iterV_tops, iterV_bottoms)
    loss = np.mean(np.array(gvl)) + np.mean(np.array(dvl)) + np.mean(np.array(fid_scores))
    return {'loss': loss, 'model': gan_bottoms, 'hyper': hyperparams, 'status': STATUS_OK}


trials_bottoms = Trials()
best_bottoms = fmin(fn=objective_bottoms, space=hyper_space, algo=tpe.suggest, max_evals=5, trials=trials_bottoms,
                    verbose=True, show_progressbar=False)

gan_bottoms = save_and_plot(trials_bottoms, iterT_tops, iterT_bottoms, iterV_tops, iterV_bottoms, 'bottoms')


# shoes in training
def objective_shoes(hyperparams):
    print(f"Tuning for shoes: {hyperparams} ")

    generatorS, discriminatorS, optimizerGS, optimizerDS = initialize(hyperparams)

    gan_shoes = SimpleGAN(device, 'shoes', generatorS, discriminatorS, optimizerGS, optimizerDS, lmd=hyperparams['lmd'])
    (_, gvl, _, dvl, fid_scores) = gan_shoes.fit(200, iterT_tops, iterT_shoes, iterV_tops, iterV_shoes)
    loss = np.mean(np.array(gvl)) + np.mean(np.array(dvl)) + np.mean(np.array(fid_scores))
    return {'loss': loss, 'model': gan_shoes, 'hyper': hyperparams, 'status': STATUS_OK}


trials_shoes = Trials()
best_shoes = fmin(fn=objective_shoes, space=hyper_space, algo=tpe.suggest, max_evals=5, trials=trials_shoes,
                  verbose=True, show_progressbar=False)

gan_shoes = save_and_plot(trials_shoes, iterT_tops, iterT_shoes, iterV_tops, iterV_shoes, 'shoes')

