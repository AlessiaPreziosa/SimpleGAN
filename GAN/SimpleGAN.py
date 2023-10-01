import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from utilities.architecture import *


class SimpleGAN:
    """
    This class creates the GAN network for each pair (tops-bottoms; tops-shoes; tops-accessories)
    of clothes in the outfit.
    """

    def __init__(self, device, item, generator, discriminator, optimizerG, optimizerD, lmd):

        super(SimpleGAN, self).__init__()
        self.device = device
        self.item = item
        self.lmd = lmd

        # define generator and discriminator of the net
        self.generator = generator
        self.discriminator = discriminator

        # move the models to GPU if available
        self.discriminator, self.generator = self.discriminator.to(device), self.generator.to(device)

        # functions for optimizers of Discriminator and Generator
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD

        # function for FID score
        self.fid = FrechetInceptionDistance(feature=64, normalize=True)
        # self.fid.set_dtype(torch.float64)  # for stability

        # functions for loss
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.L1Loss = nn.L1Loss()

    def Gloss(self, logitsD, generated_images, target_images):
        loss = (self.BCELoss(logitsD, torch.ones_like(logitsD)) +
                self.lmd * self.L1Loss(generated_images, target_images))
        return loss

    def Dloss(self, logitsD_Real, logitsD_Fake):
        loss = 0.5 * (self.BCELoss(logitsD_Real, torch.ones_like(logitsD_Real)) +
                      self.BCELoss(logitsD_Fake, torch.zeros_like(logitsD_Fake)))
        return loss

    def step(self, input_, iter_Z, step):

        param = []
        actual_DLoss, actual_GLoss, fid_score = 0.0, 0.0, 0.0
        i = 1

        for X, _ in input_:
            print('Batch:', i, 'of', len(input_))
            size = X.shape[0]
            Z, _ = next(iter_Z)
            X, Z = X.to(self.device), Z.to(self.device)

            if step == 'train':
                print('Training Discriminator...')
                # training Discriminator
                self.optimizerD.zero_grad()

            # generate images
            generated_images = self.generator(X)

            # Gradient is, for now, not computed for Generator: .detach()
            logitsD_Fake = self.discriminator(generated_images.detach(), X)

            logitsD_Real = self.discriminator(Z, X)

            lossD = self.Dloss(logitsD_Real, logitsD_Fake)
            if step == 'train':
                # updating parameters for D
                lossD.backward()
                self.optimizerD.step()

            actual_DLoss += lossD.item() * size

            if step == 'train':
                print('Training Generator...')
                # training Generator
                self.optimizerG.zero_grad()
                # Discriminator's parameters have been updated
                logitsD = self.discriminator(generated_images, X)
            else:
                logitsD = logitsD_Fake

            lossG = self.Gloss(logitsD, generated_images, Z)

            if step == 'train':
                # updating parameters for G
                lossG.backward()
                self.optimizerG.step()

            actual_GLoss += lossG.item() * size

            if step in ['validation', 'test']:
                self.fid.update(Z.to('cpu'), real=True)
                self.fid.update(generated_images.to('cpu'), real=False)
                fid = self.fid.compute()
                fid_score += fid * size
                self.fid.reset()

            i += 1

        param.append(actual_GLoss / len(input_.sampler))
        param.append(actual_DLoss / len(input_.sampler))
        if step in ['validation', 'test']:
            param.append(fid_score / len(input_.sampler))

        return param

    def fit(self, epochs, input_, target, input_val, target_val):

        generator_train_loss, discriminator_train_loss = [], []
        generator_valid_loss, discriminator_valid_loss = [], []

        fid_scores = []

        for epoch in range(epochs):

            iter_Z = iter(target)
            iter_val_Z = iter(target_val)

            print('#############################')

            print('Epoch:', epoch+1, 'of', epochs)

            print('Training...')
            # training phase
            self.generator.train()
            self.discriminator.train()

            [train_GLoss, train_DLoss] = self.step(input_, iter_Z, 'train')

            generator_train_loss.append(train_GLoss)
            discriminator_train_loss.append(train_DLoss)

            print('Validating...')
            # validation phase
            self.generator.eval()
            self.discriminator.eval()

            [valid_GLoss, valid_DLoss, fid_score] = self.step(input_val, iter_val_Z, 'validation')

            generator_valid_loss.append(valid_GLoss)
            discriminator_valid_loss.append(valid_DLoss)

            fid_scores.append(fid_score)

            if epoch % int(epochs/5) == 0:
                fig = plt.figure(figsize=(20, 20))

                imgs = []
                for X, _ in input_val:
                    X = X.to(self.device)
                    X = X[0:20, :, :, :]
                    X = self.generator(X)
                    X = X.to('cpu')
                    imgs = X.permute(0, 2, 3, 1) / 2 + 0.5
                    break

                for idx in np.arange(20):
                    fig.add_subplot(4, 5, idx + 1)
                    plt.imshow(imgs[idx].detach().numpy())

                plt.show()

        # plotting
        plot(generator_train_loss, generator_valid_loss, f'Generator: {self.item}')
        plot(discriminator_train_loss, discriminator_valid_loss, f'Discriminator: {self.item}')

        plt.plot(fid_scores)
        plt.title(f'FID score: {self.item}')
        plt.grid()
        plt.show()

        return (generator_train_loss, generator_valid_loss,
                discriminator_train_loss, discriminator_valid_loss, fid_scores)
