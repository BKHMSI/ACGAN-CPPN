import yaml 
import imageio
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Activation, Add, Lambda, Concatenate, Embedding, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, UpSampling2D, BatchNormalization, ZeroPadding2D
from keras.optimizers import Adam
from keras import backend as K

from dataloader import Dataloader

class ACGAN_CPPN:
    def __init__(self, config):
        self.z_dim = config["train"]["z-dim"] 
        self.c_dim = config["train"]["c-dim"]
        self.x_dim = config["train"]["x-dim"]
        self.y_dim = config["train"]["y-dim"]  
        self.scale = config["train"]["scale"]

        self.imchannels = config["train"]["imchannels"]
        self.imshape = (self.x_dim, self.y_dim, self.imchannels)

        self.dataloader = Dataloader(config)
        self.num_classes = config["train"]["num-classes"]

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator(self.x_dim, self.y_dim)

        self.combined = self.build_combined()
        self.combined.compile(loss=losses, optimizer=optimizer)

    def build_combined(self):

        # latent vector
        z = Input(shape=(self.z_dim,))
        # inputs to cppn, like coordinates and radius from centre
        x = Input(shape=(None, 1))
        y = Input(shape=(None, 1))
        r = Input(shape=(None, 1))
        # input condition
        c = Input(shape=(1,), dtype='int32')

        image = self.generator([z, c, x, y, r])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(image)

        # The combined model (stacked generator and discriminator)
        return Model([z, c, x, y, r], valid)
        
    def build_generator(self, x_dim, y_dim, net_size=256):

        # latent vector
        z = Input(shape=(self.z_dim,))
        # inputs to cppn, like coordinates and radius from centre
        x = Input(shape=(None, 1))
        y = Input(shape=(None, 1))
        r = Input(shape=(None, 1))
        # input condition
        c = Input(shape=(1,), dtype='int32')

        n_points = x_dim * y_dim

        z_reshaped = Reshape((1, self.z_dim))(z)
        z_scaled   = Lambda(lambda ze: ze * K.ones((n_points,1), dtype="float32") * self.scale)(z_reshaped)

        c_embedding = Embedding(self.num_classes, self.c_dim)(c)
        c_scaled = Lambda(lambda ce: ce * K.ones((n_points,1), dtype="float32") * self.scale)(c_embedding)

        z_unroll = Reshape((n_points, self.z_dim))(z_scaled)
        c_unroll = Reshape((n_points, self.c_dim))(c_scaled)
        x_unroll = Reshape((n_points, 1))(x)
        y_unroll = Reshape((n_points, 1))(y)
        r_unroll = Reshape((n_points, 1))(r)

        u0 = Dense(net_size)(z_unroll)
        u1 = Dense(net_size)(x_unroll)
        u2 = Dense(net_size)(y_unroll)
        u3 = Dense(net_size)(r_unroll)
        u4 = Dense(net_size)(c_unroll)

        h = Concatenate()([u0, u1, u2, u3, u4])

        for _ in range(5):
            h = Dense(net_size, activation="relu")(h)

        output = Dense(self.imchannels, activation="tanh")(h)
        result = Reshape((x_dim, y_dim, self.imchannels))(output)
        
        return Model([z, c, x, y, r], result)

    def build_discriminator(self):
        image = Input(shape=(self.imshape))

        x = Conv2D(16, kernel_size=3, strides=2, padding="same")(image)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
        x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = BatchNormalization(momentum=0.8)(x)
    
        x = Conv2D(128, kernel_size=3, strides=1, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)

        validity = Dense(1, activation="sigmoid")(x)
        label = Dense(self.num_classes, activation="softmax")(x)

        return Model(image, [validity, label])

    def train(self, batch_size, epochs, save_interval):
        # Load the dataset
        # (X_train, y_train), (_, _) = mnist.load_data()

        # # Invert images
        # X_train = 255 - X_train
        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        x_vec, y_vec, r_vec = self._coordinates(batch_size, self.x_dim, self.y_dim, scale = self.scale)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # images = X_train[idx]
            # labels = y_train[idx]

            images, labels = self.dataloader.sample(batch_size)

            sampled_labels = np.random.randint(0, self.num_classes, (batch_size, 1))
            z = np.random.normal(0, 1, size=(batch_size, self.z_dim)).astype(np.float32)
            gen_images = self.generator.predict([z, sampled_labels, x_vec, y_vec, r_vec])

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(images, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_images, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([z, sampled_labels, x_vec, y_vec, r_vec], [valid, sampled_labels])
        
            # Plot the progress
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]}, acc.: {d_loss[3]*100:.2f}, {d_loss[4]*100:.2f}] [G loss: {g_loss[0]}]", end="\r")

            if (epoch+1) % save_interval == 0:
                self.generator.save_weights(f"weights/qd-cgenerator-weights.{epoch+1}.h5")
                self.sample_images(epoch+1)

    def generate(self, batch_size, z=None, labels=None, x_dim = 28, y_dim = 28, scale = 8.0):
        """ Generate data by sampling from latent space.
        If z is not None, data for this point in latent space is
        generated. Otherwise, z is drawn from prior in latent
        space.
        """
        if z is None:
            z = np.random.normal(0, 1, size=(batch_size, self.z_dim)).astype(np.float32)

        if labels is None:
            labels = np.random.randint(0, self.num_classes, (batch_size, 1))

        x_vec, y_vec, r_vec = self._coordinates(batch_size, x_dim, y_dim, scale = scale)
        images = self.generator.predict([z, labels, x_vec, y_vec, r_vec])
        return images

    def sample_images(self, epoch):
        r, c = 3, 3

        sampled_labels = np.random.randint(0, self.num_classes, (r*c, 1))
        gen_images = self.generate(r*c, labels=sampled_labels)
        
        # Rescale images 0 - 1
        gen_images = 0.5 * gen_images + 0.5

        fig, axs = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_images[i*c+j].squeeze(), cmap='gray')
                axs[i,j].set_title(f"{self.dataloader.labels[sampled_labels[i*c+j][0]]}")
                axs[i,j].axis('off')
        fig.savefig(f"samples/sample-{epoch}.png")
        plt.close()

    def generate_high_resolution(self, config):
        self.generator = self.build_generator(config["generate"]["x-dim"], config["generate"]["y-dim"])
        self.generator.load_weights(config["generate"]["weights-path"])

        images = self.generate(config["generate"]["n-samples"], x_dim=config["generate"]["x-dim"], y_dim=config["generate"]["y-dim"])
        images = 0.5 * images + 0.5
        cmap = "gray" if config["train"]["imchannels"] == 1 else None
        for image in images:
            plt.imshow(image.squeeze(), cmap=cmap)
            plt.show()

    def _coordinates(self, batch_size, x_dim = 28, y_dim = 28, scale = 1.0):
        '''
        calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
        '''
        n_points = x_dim * y_dim
        x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
        y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
        y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
        r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
        return x_mat, y_mat, r_mat

    def latent_interpolation(self, config, n_steps, n_frames):

        self.generator = self.build_generator(config["generate"]["x-dim"], config["generate"]["y-dim"])
        self.generator.load_weights(config["generate"]["weights-path"])

        rand_label = np.random.choice(np.arange(self.num_classes), n_frames, replace=False)
        rand_label = np.arange(10)

        labels = []
        for label in rand_label: labels.extend([label]*n_steps)
        labels = np.array(labels)

        vectors = []
        alpha_values = np.linspace(0, 1, n_steps)

        start_vec = np.random.normal(0, 1, (self.z_dim))

        for _ in range(n_frames):
            end_vec = np.random.normal(0, 1, (self.z_dim))

            for alpha in alpha_values:
                vector = start_vec*(1-alpha) + end_vec*alpha
                vectors.append(vector)

            start_vec = end_vec

        images = np.zeros((len(labels), config["generate"]["x-dim"], config["generate"]["y-dim"], config["train"]["imchannels"]))
        for i, vector in enumerate(vectors):
            images[i] = self.generate(1, z=np.array([vector]), x_dim=config["generate"]["x-dim"], y_dim=config["generate"]["y-dim"])
        
        images = 0.5 * images + 0.5
        images = np.uint8(images * 255)
        imageio.imsave("sample.gif", images)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CPPN Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)


    model = ACGAN_CPPN(config)
    model.train(config["train"]["batch-size"], config["train"]["epochs"], config["train"]["save-interval"])
    # model.generate_high_resolution(config)
    # model.latent_interpolation(config, 10, 10)

