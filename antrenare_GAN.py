import sys, random, os # manipulare sistem, val. aleatorii, lucru cu fisiere
import numpy as np # manipularea datelor numerice
from matplotlib import pyplot as plt # vizualizare date
import pydot # creare si manipulare grafice
import cv2 # biblioteca OpenCV pentru procesarea imaginilor

# import quantize

NUM_EPOCHS = 2000 # nr epoci

LR_D = 0.0004 # rata invatare discriminator
LR_G = 0.001 # rata invatare generator
BETA_1 = 0.8 # pt. optimizatorul Adam, controleaza viteza de adaptare a ratelor de invatare
EPSILON = 1e-4 # pt. stabilitate numerica in optimizare
ENC_WEIGHT = 200.0 # greutate asociata pierderii encoderului in Loss-functionul total
BN_M = 0.8 # Batch Normalization Momentum
DO_RATE = 0.25 # drop-out rate-ul neuronii sunt dezactivati cu o prob. de 0.25 in timpul antrenarii
NOISE_SIGMA = 0.15 # deviatia standard a zgomotului pt. datele de intrare
CONTINUE_TRAIN = False
NUM_RAND_COMICS = 10 # nr. benzi generate aleatoriu pentru vizualizare
BATCH_SIZE = 5 # dimensiune lot date pentru antrenare
PARAM_SIZE = 160 # dimensiunea vectorului de zgomot pentru generator
COMIC_PARAMS = 320 # dimensiunea param asociati unui set de date

PREV_V = None 
means = None # media datelor
# Valori si vectori proprii pt. analiza spectrala
evals = None
evecs = None


def plotScores(scores, fname, on_top=True):
    """
    Functie ce ploteaza si salveaza graficele ce arata evolutia 
    scorurilor pe epoci.
    
    """
    plt.clf()
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.grid(True)
    for s in scores:
        plt.plot(s)
    plt.xlabel('Epoch')
    loc = ('upper right' if on_top else 'lower right')
    plt.legend(['Dis', 'Gen', 'Enc'], loc=loc)
    plt.draw()
    plt.savefig(fname)


def save_config():
    """
    Functie ce salveaza configuratia retelei intr-un
    fisier text.
    
    """
    with open('config.txt', 'w') as fout:
        fout.write('LR_D:        ' + str(LR_D) + '\n')
        fout.write('LR_G:        ' + str(LR_G) + '\n')
        fout.write('BETA_1:      ' + str(BETA_1) + '\n')
        fout.write('BN_M:        ' + str(BN_M) + '\n')
        fout.write('BATCH_SIZE:  ' + str(BATCH_SIZE) + '\n')
        fout.write('DO_RATE:     ' + str(DO_RATE) + '\n')
        fout.write('NOISE_SIGMA: ' + str(NOISE_SIGMA) + '\n')
        fout.write('EPSILON:     ' + str(EPSILON) + '\n')
        fout.write('ENC_WEIGHT:  ' + str(ENC_WEIGHT) + '\n')
        fout.write('optimizer_d: ' + type(d_optimizer).__name__ + '\n')
        fout.write('optimizer_g: ' + type(g_optimizer).__name__ + '\n')


def to_comic(fname, x):
    """
    Transforma un array de imagini intr-o imagine de tip
    banda desenata si o salveaza intr-un fisier.
    
    """

    # conversie din array din intervalul [0, 1] in [0, 255]
    # si transformare in tipul unit 8
    img = (x * 255.0).astype(np.uint8)

    # imaginea are 4 dimensiuni [batch, height, width, channels]
    # combinam imaginile pe axa canalelor pentru a obtine o imagine mare
    if len(img.shape) == 4:
        img = np.concatenate(img, axis=2)

    # reordonam dimensiunea imaginii 
    # inversam ordinea pe exa verticala
    # transpunem pentru a se potrivi cu formatul OpenCV (height, width, channels)
    img = np.transpose(img[::-1], (1, 2, 0))

    # salvam
    cv2.imwrite(fname, img)
    if fname == 'rand0.png':
        cv2.imwrite('rand0/r' + str(iters) + '.png', img)


def make_rand_comics(write_dir, rand_vecs):

    """
    Genereaza benzi desenate aleatorii folosind vectori random si le salveaza

    rand_vecs :  matrice de vectori random folositi pentru zgomot de intrare pe generator
    
    """
    y_comics = generator.predict(rand_vecs)
    for i in range(rand_vecs.shape[0]):
        to_comic('rand' + str(i) + '.png', y_comics[i])


def make_rand_comics_normalized(write_dir, rand_vecs):

    """
    Fata de functia de mai sus salveaza si statistici si grafice
    
    """
    global PREV_V # Vect. proprii din epoca anterioara
    global means # media datelor de codificare

    # valorile si vectorii proprii obtinuti din PCA
    global evals
    global evecs

    # codifica mostrele de date prin encoderul GAN-ului
    x_enc = np.squeeze(encoder.predict(x_samples))

    # media, deviatia standard si matricea de covarianta
    means = np.mean(x_enc, axis=0)
    x_stds = np.std(x_enc, axis=0)
    x_cov = np.cov((x_enc - means).T)

    # desc. in val. proprii si valori singulare
    u, s, evecs = np.linalg.svd(x_cov)
    evals = np.sqrt(s)

    # This step is not necessary, but it makes the random generated test
    # samples consistent between epochs so you can see the evolution of
    # the training better.
    #
    # Like square roots, each prinicpal component has 2 solutions that
    # represent opposing vector directions.  For each component, just
    # choose the direction that was closest to the last epoch.

    if PREV_V is not None:
        d = np.sum(PREV_V * evecs, axis=1) # prod. scalar intre vect. proprii vechi si noi
        d = np.where(d > 0.0, 1.0, -1.0) # decidem directia vectorilor proprii
        evecs = evecs * np.expand_dims(d, axis=1)
    PREV_V = evecs # salvam p. urm epoca

    print("Evals: ", evals[:6]) # primele 6 valori proprii

    # Salvam
    np.save(write_dir + 'means.npy', means)
    np.save(write_dir + 'stds.npy', x_stds)
    np.save(write_dir + 'evals.npy', evals)
    np.save(write_dir + 'evecs.npy', evecs)

    # Creem noi benzi desenate
    x_vecs = means + np.dot(rand_vecs * evals, evecs)
    make_rand_comics(write_dir, x_vecs)

    # Creem titlu pt. grafice daca directorul include informatii despre epoca
    title = ''
    if '/' in write_dir:
        title = 'Epoch: ' + write_dir.split('/')[-2][1:]

    # Graficele:

    plt.clf()
    plt.title(title)
    plt.bar(np.arange(evals.shape[0]), evals, align='center')
    plt.draw()
    plt.savefig(write_dir + 'evals.png')

    plt.clf()
    plt.title(title)
    plt.bar(np.arange(means.shape[0]), means, align='center')
    plt.draw()
    plt.savefig(write_dir + 'means.png')

    plt.clf()
    plt.title(title)
    plt.bar(np.arange(x_stds.shape[0]), x_stds, align='center')
    plt.draw()
    plt.savefig(write_dir + 'stds.png')


def save_models():
    """
    Salvam modelele in fisiere .h5
    """
    discriminator.save('discriminator.h5')
    generator.save('generator.h5')
    encoder.save('encoder.h5')
    print("Saved")


###################################
#  Load Keras
###################################
print("Loading Keras...")

# Import Biblioneci pt. Keras si tensorflow

import os, math
import tensorflow.keras

print("Keras Version: " + tensorflow.keras.__version__)
from tensorflow.keras.layers import (
    Input, Dense, Activation, Dropout, Flatten, Reshape, Permute, RepeatVector,
    ActivityRegularization, TimeDistributed, Lambda, LeakyReLU, Conv1D, Conv2D,
    Conv2DTranspose, UpSampling2D, ZeroPadding2D, Embedding, MaxPooling2D,
    AveragePooling2D, GaussianNoise, BatchNormalization, LSTM, SimpleRNN
)
from tensorflow.keras.initializers import RandomNormal # Initializar distribuitia normala
from tensorflow.keras.losses import binary_crossentropy # Loss `function pt. clasificare binara`
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD # Optimizatori pt. antrenare
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Generator Imagini
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model # Pt. Vizualizarea Modelului
from tensorflow.keras.activations import softmax
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.layers import Layer  # Replaces keras.engine.topology.Layer

# Selecatrea formatului datelor imagistice ca channel fistt (canalele de culoare sunt pe prima dimensiune)
K.set_image_data_format('channels_first')

# Creeaza un set de vectori random normal distribuiti (zgomot) pt. testare  de dimensiune NUM_RAND_COMICS x PARAM_SIZE
np.random.seed(0)
random.seed(0)
z_test = np.random.normal(0.0, 1.0, (NUM_RAND_COMICS, PARAM_SIZE))

###################################
#  Load Dataset
###################################
print("Loading Data...")

# Incarcam setul de date cu imagini din fisierul Numpy cu benzile desenate preprocesate
y_samples = np.load('top10000.npy')
y_shape = y_samples.shape
num_samples = y_samples.shape[0]

# Creem intrari numerice (index-uri) pt. setul de date
# x_samples este de orma [ [0], [1], ..., [num_samples - 1]]
x_samples = np.expand_dims(np.arange(num_samples), axis=1)

x_shape = x_samples.shape # dimensiunea lui x_samples
z_shape = (PARAM_SIZE,) # dimenisunea vectorului de zgomot

print("Loaded " + str(num_samples) + " panels.")

y_test = y_samples[0].astype(np.float32) / 255.0 # normalizare pe [0, 1]
x_test = np.copy(x_samples[0:1])

###################################
#  Create Model
###################################
if CONTINUE_TRAIN:
    # Daca modelele sunt deja create incarcam din fisierele .h5

    print("Loading Discriminator...")
    discriminator = load_model('discriminator.h5')
    print("Loading Generator...")
    generator = load_model('generator.h5')
    print("Loading Encoder...")
    encoder = load_model('encoder.h5')
    print("Loading Vectors...")
    PREV_V = np.load('evecs.npy')
    z_test = np.load('rand.npy')
else:
    print("Building Discriminator...")

    input_shape = y_shape[1:] # forma de inrare a datelor excludem prima dimensiune lotul
    print(input_shape)

    discriminator = Sequential() # model secvential

    # 1. Adaugam zgomot Gaussian pt. augmentarea datelor de intrare
    discriminator.add(GaussianNoise(NOISE_SIGMA, input_shape=input_shape))

    # 2. Primul strat convolutional cu 40 de filtre
    discriminator.add(TimeDistributed(Conv2D(40, (5, 5), padding='same')))
    discriminator.add(LeakyReLU(0.2)) # functie de activare
    discriminator.add(TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))) # normalizare pt. stabilitate
    if DO_RATE > 0:
        discriminator.add(Dropout(DO_RATE))
    print(discriminator.output_shape)

    # 3. MaxPooling pt reducerea dimensionalitatii
    discriminator.add(TimeDistributed(MaxPooling2D(4)))
    print(discriminator.output_shape)

    # 4. Al doilea strat convolutional cu 80 de filtre
    discriminator.add(TimeDistributed(Conv2D(80, (5, 5), padding='same')))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(TimeDistributed(BatchNormalization(momentum=BN_M, axis=1)))
    if DO_RATE > 0:
        discriminator.add(Dropout(DO_RATE))
    print(discriminator.output_shape)

    # 5. MaxPooling pt. refucerea dimensiunilor
    discriminator.add(TimeDistributed(MaxPooling2D(4)))
    print(discriminator.output_shape)

    # 6. Al treilea strat convolutional cu 120 de filtre
    discriminator.add(TimeDistributed(Conv2D(120, (5, 5), padding='same')))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(TimeDistributed(BatchNormalization(momentum=BN_M, axis=1)))
    if DO_RATE > 0:
        discriminator.add(Dropout(DO_RATE))
    print(discriminator.output_shape)

    # 7. MaxPooling pt. refucerea dimensiunilor
    discriminator.add(TimeDistributed(MaxPooling2D(8)))
    print(discriminator.output_shape)

    # 8. Aplatizare iesirii pt. a trece la straturi dense
    discriminator.add(Flatten(data_format='channels_last'))
    print(discriminator.output_shape)

    # 9. Strat dens pentru clasificarea binara (1 neuron activat cu sigmoid) (e banda desenata reala sau falsa)
    discriminator.add(Dense(1, activation='sigmoid'))
    print(discriminator.output_shape)

    print("Building Generator...")

    generator = Sequential() # Model Segvential
    input_shape = (PARAM_SIZE,) # dimensiunea intrarii e un vector aleator (zgomot)
    print(input_shape)

    # 1. primul Strat dens cu 600 de neuroni 
    generator.add(Dense(600, input_shape=input_shape))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=BN_M))
    print(generator.output_shape)

    # 2. al 2-lea strat dens de dimeniunea y_shape[1] (nr benzi desenate) * COMIC_PARAMS
    generator.add(Dense(y_shape[1] * COMIC_PARAMS))
    generator.add(LeakyReLU(0.2))
    print(generator.output_shape)

    # 3. Reshape pentru a transforma vectorul intr-o matrice de nr_benzi * comic_params
    generator.add(Reshape((y_shape[1], COMIC_PARAMS)))
    generator.add(TimeDistributed(BatchNormalization(momentum=BN_M)))
    print(generator.output_shape)

    # 4. Strat dens pt. a mari dimensiunea la 200 * 4 * 4
    generator.add(TimeDistributed(Dense(200 * 4 * 4)))
    print(generator.output_shape)

    # 5. Reshape pentru a transforma dimenisunea in (200, 4, 4)
    generator.add(Reshape((y_shape[1], 200, 4, 4)))
    generator.add(LeakyReLU(0.2))
    if DO_RATE > 0:
        generator.add(Dropout(DO_RATE))
    # generator.add(BatchNormalization(momentum=BN_M, axis=1))
    print(generator.output_shape)

    # 6. Conv2DTranspose: exitinde dimensiunea spatiala prin operatii inverse convolutiei pt a genera imagini
    # Vom utiliza 6 astfel de straturi
    generator.add(TimeDistributed(Conv2DTranspose(200, (5, 5), strides=(2, 2), padding='same')))
    generator.add(LeakyReLU(0.2))
    if DO_RATE > 0:
        generator.add(Dropout(DO_RATE))
    # generator.add(BatchNormalization(momentum=BN_M, axis=1))
    print(generator.output_shape)

    generator.add(TimeDistributed(Conv2DTranspose(160, (5, 5), strides=(2, 2), padding='same')))
    generator.add(LeakyReLU(0.2))
    if DO_RATE > 0:
        generator.add(Dropout(DO_RATE))
    # generator.add(BatchNormalization(momentum=BN_M, axis=1))
    print(generator.output_shape)

    generator.add(TimeDistributed(Conv2DTranspose(120, (5, 5), strides=(2, 2), padding='same')))
    generator.add(LeakyReLU(0.2))
    if DO_RATE > 0:
        generator.add(Dropout(DO_RATE))
    # generator.add(BatchNormalization(momentum=BN_M, axis=1))
    print(generator.output_shape)

    generator.add(TimeDistributed(Conv2DTranspose(80, (5, 5), strides=(2, 2), padding='same')))
    generator.add(LeakyReLU(0.2))
    if DO_RATE > 0:
        generator.add(Dropout(DO_RATE))
    # generator.add(BatchNormalization(momentum=BN_M, axis=1))
    print(generator.output_shape)

    generator.add(TimeDistributed(Conv2DTranspose(40, (5, 5), strides=(2, 2), padding='same')))
    generator.add(LeakyReLU(0.2))
    if DO_RATE > 0:
        generator.add(Dropout(DO_RATE))
    # generator.add(BatchNormalization(momentum=BN_M, axis=1))
    print(generator.output_shape)

    generator.add(TimeDistributed(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='sigmoid')))
    print(generator.output_shape)

    print("Building Encoder...")
    """
    ENCODERUL este utilizat aici pentru a invata reprezentarea latenta compacta a datelor de intrare
    
    """
    encoder = Sequential()
    print(num_samples)
    """
    Embedding-ul:

    transforma indexurile numerice de la x_samples in vectori densi intr-un spatiu latent
    folosit pentru a invata o reprezentare continua a datelor discrete
    
    imput_dim-ul = num_samples  este num_samples numarul total de intrari unice 
    output_dim-ul = PARAM_SIZE dimensiunea vect. embedding generati setata la param_size pt a fi compatibil cu zgomotul din generator
    mbeddings_initializer greutatile sunt initializate cu valori cu distributie normala cu deviatia standard 1e-4
    """
    encoder.add(Embedding(num_samples, PARAM_SIZE, embeddings_initializer=RandomNormal(stddev=1e-4)))

    # transorma un vector multidimensional in unul unidimensional pentru a cupla encoderul la intrarea altor modele
    encoder.add(Flatten(data_format='channels_last'))
    # print(encoder.output_shape)

print("Building GANN...") # Construim modelul GAN-ului

# Configurarea optimizatorului folosim Adam un optimizator avansat 
d_optimizer = Adam(learning_rate=LR_D, beta_1=BETA_1, epsilon=EPSILON)
g_optimizer = Adam(learning_rate=LR_G, beta_1=BETA_1, epsilon=EPSILON)

# MODEL DISCRIMINATOR
discriminator.trainable = True
generator.trainable = False
encoder.trainable = False

# Definim Intrarile:
d_in_real = Input(shape=y_shape[1:]) # date reale (imagini autentice)
d_in_fake = Input(shape=x_shape[1:]) # date false (zgomot)

d_fake = generator(encoder(d_in_fake)) # converteste datele false in imagini generate

# Trecerea datelor reale si generate prin discrimiator
d_out_real = discriminator(d_in_real)
d_out_real = Activation('linear', name='d_out_real')(d_out_real)
d_out_fake = discriminator(d_fake)
d_out_fake = Activation('linear', name='d_out_fake')(d_out_fake)

dis_model = Model(inputs=[d_in_real, d_in_fake], outputs=[d_out_real, d_out_fake])
dis_model.compile(
    optimizer=d_optimizer,
    loss={'d_out_real': 'binary_crossentropy', 'd_out_fake': 'binary_crossentropy'},
    loss_weights={'d_out_real': 1.0, 'd_out_fake': 1.0})


# MODEL GENERATOR-DISCRIMINATOR
discriminator.trainable = False
generator.trainable = True
encoder.trainable = True

g_in = Input(shape=x_shape[1:]) # Intrarea generatorului vector de zgomot
g_enc = encoder(g_in) # Codificare prin encoder

g_out_img = generator(g_enc) # Imaginea Generata
g_out_img = Activation('linear', name='g_out_img')(g_out_img)
g_out_dis = discriminator(g_out_img)
g_out_dis = Activation('linear', name='g_out_dis')(g_out_dis)

gen_dis_model = Model(inputs=[g_in], outputs=[g_out_img, g_out_dis])
gen_dis_model.compile(
    optimizer=g_optimizer,
    loss={'g_out_img': 'mse', 'g_out_dis': 'binary_crossentropy'},
    loss_weights={'g_out_img': ENC_WEIGHT, 'g_out_dis': 1.0})

plot_model(gen_dis_model, to_file='generator.png', show_shapes=True)
plot_model(dis_model, to_file='discriminator.png', show_shapes=True)

###################################
#  Train
###################################

# BUCLA DE ANTRENARE GAN

# salvam vectorii aleatori de test, imaginile de referinta si modelele
np.save('rand.npy', z_test)
to_comic('gt.png', y_test)
save_models()

print("Training...")
save_config()
# Liste de Loss pntru fiecare model
generator_loss = []
discriminator_loss = []
encoder_loss = []

# Ones, Zeros sunt vectori de etichete de dim. Numsaples folositi ca etichete pt antrenarea discriminatorului
ones = np.ones((num_samples,), dtype=np.float32)
zeros = np.zeros((num_samples,), dtype=np.float32)

iters = 0
""" 
genereaza si salveaza benzi desenate folosind vectorii aleatori z_test 
inainte de inceperea antrenarii  pentru a avea un punct de referinta.
"""
make_rand_comics('', z_test)

for iters in range(NUM_EPOCHS):
    loss_d = 0.0
    loss_g = 0.0
    loss_e = 0.0
    num_d = 0
    num_g = 0
    num_e = 0

    ratio_g = 1 # raportul de cat de des e antrenat generatorul si discriminatorul in cazul nostru 1 la 1
    np.random.shuffle(x_samples) # adaugam randomness in x_samples pt a ne asigura ca modelul nu invata dupa un tipar
    for i in range(0, num_samples // BATCH_SIZE):
        if i % ratio_g == 0: # conditie antrenare discriminator
            # Make samples
            j = i // ratio_g
            # LOTUL PT DISCRIMINATOR
            x_batch1 = x_samples[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            y_batch1 = y_samples[x_batch1[:, 0]].astype(np.float32) / 255.0

            ones = np.ones((BATCH_SIZE,), dtype=np.float32)
            zeros = np.zeros((BATCH_SIZE,), dtype=np.float32)

            losses = dis_model.train_on_batch([y_batch1, x_batch1], [ones, zeros])
            names = dis_model.metrics_names
            loss_d += losses[names.index('d_out_real_loss')]
            loss_d += losses[names.index('d_out_fake_loss')]
            num_d += 2
        # LOTUL PT DISCRIMINATOR SI ENCODER
        x_batch2 = x_samples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        y_batch2 = y_samples[x_batch2[:, 0]].astype(np.float32) / 255.0

        losses = gen_dis_model.train_on_batch([x_batch2], [y_batch2, ones])
        names = gen_dis_model.metrics_names
        
        loss_e += losses[names.index('g_out_img_loss')]
        num_e += 1
        print(loss_e)
        print(num_e)
        loss_g += losses[names.index('g_out_dis_loss')]
        num_g += 1
        
        progress = (i * 100) * BATCH_SIZE / num_samples
        sys.stdout.write(
            str(progress) + "%" +
            "  D:" + str(loss_d / num_d) +
            "  G:" + str(loss_g / num_g) +
            "  E:" + str(loss_e / num_e) + "        ")
        sys.stdout.write('\r')
        sys.stdout.flush()
    sys.stdout.write('\n')

    discriminator_loss.append(loss_d / num_d)
    generator_loss.append(loss_g / num_g)
    encoder_loss.append(loss_e * 10.0 / num_e)

    plotScores([discriminator_loss, generator_loss, encoder_loss], 'Scores.png')

    save_models()

    # Generate some random comics
    y_enc = encoder.predict(x_test, batch_size=1)
    y_comic = generator.predict(y_enc, batch_size=1)[0]
    to_comic('test.png', y_comic)
    make_rand_comics('', z_test)

print("Done")
