import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from time import time
from PIL import Image
import glob
import re
from struct import *
from skimage.transform import resize
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import SpectralEmbedding as LaplacianEigenmap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from supervised_PCA import Supervised_PCA as SPCA
from sklearn.manifold import TSNE
from metric_learning_closed_form_efficient import Metric_learning_closed_form as ML
from kernel_FLDA_efficient import Kernel_FLDA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB as NB


def main():
    # ----- settings:
    dataset = 'MNIST'    # --> 'Facial' or 'MNIST' or 'Breast_cancer'
    embedding_method = 'Isomap'
    n_components = 5
    split_in_cross_validation_again = False
    load_dataset_again = False
    subset_of_MNIST = True
    pick_subset_of_MNIST_again = False
    MNIST_subset_cardinality_training = 10000   # picking from first samples of 60,000 samples
    MNIST_subset_cardinality_testing = 5000  # picking from first samples of 10,000 samples
    # ----- paths:
    if dataset == 'Facial':
        path_dataset = './input/att_database/'
        path_dataset_save = './input/pickle_dataset/Facial/'
    elif dataset == 'MNIST':
        path_dataset = './input/mnist/'
        path_dataset_save = './input/pickle_dataset/MNIST/'
    elif dataset == 'Breast_cancer':
        path_dataset = './input/Breast_cancer_dataset/wdbc_data.txt'
        path_dataset_save = './input/pickle_dataset/MNIST/'
    # ----- Loading dataset:
    print('Reading dataset...')
    if dataset == 'MNIST':
        if load_dataset_again:
            training_data = list(read_MNIST_dataset(dataset = "training", path = path_dataset))
            testing_data = list(read_MNIST_dataset(dataset = "testing", path = path_dataset))

            number_of_training_samples = len(training_data)
            dimension_of_data = 28 * 28
            X_train = np.empty((0, dimension_of_data))
            y_train = np.empty((0, 1))
            for sample_index in range(number_of_training_samples):
                if np.mod(sample_index, 1) == 0:
                    print('sample ' + str(sample_index) + ' from ' + str(number_of_training_samples) + ' samples...')
                label, pixels = training_data[sample_index]
                pixels_reshaped = np.reshape(pixels, (1, 28*28))
                X_train = np.vstack([X_train, pixels_reshaped])
                y_train = np.vstack([y_train, label])
            y_train = y_train.ravel()

            number_of_testing_samples = len(testing_data)
            dimension_of_data = 28 * 28
            X_test = np.empty((0, dimension_of_data))
            y_test = np.empty((0, 1))
            for sample_index in range(number_of_testing_samples):
                if np.mod(sample_index, 1) == 0:
                    print('sample ' + str(sample_index) + ' from ' + str(number_of_testing_samples) + ' samples...')
                label, pixels = testing_data[sample_index]
                pixels_reshaped = np.reshape(pixels, (1, 28*28))
                X_test = np.vstack([X_test, pixels_reshaped])
                y_test = np.vstack([y_test, label])
            y_test = y_test.ravel()

            save_variable(X_train, 'X_train', path_to_save=path_dataset_save)
            save_variable(y_train, 'y_train', path_to_save=path_dataset_save)
            save_variable(X_test, 'X_test', path_to_save=path_dataset_save)
            save_variable(y_test, 'y_test', path_to_save=path_dataset_save)
        else:
            file = open(path_dataset_save+'X_train.pckl','rb')
            X_train = pickle.load(file); file.close()
            file = open(path_dataset_save+'y_train.pckl','rb')
            y_train = pickle.load(file); file.close()
            file = open(path_dataset_save+'X_test.pckl','rb')
            X_test = pickle.load(file); file.close()
            file = open(path_dataset_save+'y_test.pckl','rb')
            y_test = pickle.load(file); file.close()

        if subset_of_MNIST:
            if pick_subset_of_MNIST_again:
                X_train_picked = X_train[0:MNIST_subset_cardinality_training, :]
                X_test_picked = X_test[0:MNIST_subset_cardinality_testing, :]
                y_train_picked = y_train[0:MNIST_subset_cardinality_training]
                y_test_picked = y_test[0:MNIST_subset_cardinality_testing]
                save_variable(X_train_picked, 'X_train_picked', path_to_save=path_dataset_save)
                save_variable(X_test_picked, 'X_test_picked', path_to_save=path_dataset_save)
                save_variable(y_train_picked, 'y_train_picked', path_to_save=path_dataset_save)
                save_variable(y_test_picked, 'y_test_picked', path_to_save=path_dataset_save)
            else:
                file = open(path_dataset_save+'X_train_picked.pckl','rb')
                X_train_picked = pickle.load(file); file.close()
                file = open(path_dataset_save+'X_test_picked.pckl','rb')
                X_test_picked = pickle.load(file); file.close()
                file = open(path_dataset_save+'y_train_picked.pckl','rb')
                y_train_picked = pickle.load(file); file.close()
                file = open(path_dataset_save+'y_test_picked.pckl','rb')
                y_test_picked = pickle.load(file); file.close()
            X_train = X_train_picked
            X_test = X_test_picked
            y_train = y_train_picked
            y_test = y_test_picked
        image_shape = (28, 28)
    elif dataset == 'Facial':
        if load_dataset_again:
            X, y, image_shape = read_image_dataset(dataset_path=path_dataset, imagesType='.jpg')
            save_variable(variable=X, name_of_variable='X', path_to_save=path_dataset_save)
            save_variable(variable=y, name_of_variable='y', path_to_save=path_dataset_save)
            save_variable(variable=image_shape, name_of_variable='image_shape', path_to_save=path_dataset_save)
        else:
            file = open(path_dataset_save+'X.pckl','rb'); X = pickle.load(file); file.close()
            file = open(path_dataset_save+'y.pckl','rb'); y = pickle.load(file); file.close()
            file = open(path_dataset_save+'image_shape.pckl','rb'); image_shape = pickle.load(file); file.close()
    elif dataset == 'Breast_cancer':
        data = pd.read_csv(path_dataset, sep=",", header=None)  # read text file using pandas dataFrame: https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas
        labels_of_classes = ['M', 'B']
        X, y = read_BreastCancer_dataset(data=data, labels_of_classes=labels_of_classes)
        X = X.astype(np.float64)  #---> otherwise MDS has error --> https://stackoverflow.com/questions/16990996/multidimensional-scaling-fitting-in-numpy-pandas-and-sklearn-valueerror
        # --- cross validation:
        path_to_save = './input/split_data/'
        portion_of_test_in_dataset = 0.3
        number_of_folds = 10
        if split_in_cross_validation_again:
            train_indices_in_folds, test_indices_in_folds, \
            X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds = \
                cross_validation(X=X, y=y, n_splits=number_of_folds, test_size=portion_of_test_in_dataset)
            save_variable(train_indices_in_folds, 'train_indices_in_folds', path_to_save=path_to_save)
            save_variable(test_indices_in_folds, 'test_indices_in_folds', path_to_save=path_to_save)
            save_variable(X_train_in_folds, 'X_train_in_folds', path_to_save=path_to_save)
            save_variable(X_test_in_folds, 'X_test_in_folds', path_to_save=path_to_save)
            save_variable(y_train_in_folds, 'y_train_in_folds', path_to_save=path_to_save)
            save_variable(y_test_in_folds, 'y_test_in_folds', path_to_save=path_to_save)
            for fold_index in range(number_of_folds):
                save_np_array_to_txt(np.asarray(train_indices_in_folds[fold_index]), 'train_indices_in_fold' + str(fold_index), path_to_save=path_to_save)
                save_np_array_to_txt(np.asarray(test_indices_in_folds[fold_index]), 'test_indices_in_folds' + str(fold_index), path_to_save=path_to_save)
        else:
            file = open(path_to_save + 'train_indices_in_folds.pckl', 'rb')
            train_indices_in_folds = pickle.load(file); file.close()
            file = open(path_to_save + 'test_indices_in_folds.pckl', 'rb')
            test_indices_in_folds = pickle.load(file); file.close()
            file = open(path_to_save + 'X_train_in_folds.pckl', 'rb')
            X_train_in_folds = pickle.load(file); file.close()
            file = open(path_to_save + 'X_test_in_folds.pckl', 'rb')
            X_test_in_folds = pickle.load(file); file.close()
            file = open(path_to_save + 'y_train_in_folds.pckl', 'rb')
            y_train_in_folds = pickle.load(file); file.close()
            file = open(path_to_save + 'y_test_in_folds.pckl', 'rb')
            y_test_in_folds = pickle.load(file); file.close()

    print(X_train.shape)
    print(X_test.shape)

    # ----- embedding:
    print('Embedding...')
    if dataset == 'MNIST':
        # plot_components(X_projected=X_projected, images=X.reshape((-1, image_shape[0], image_shape[1])), ax=ax, image_scale=0.6, markersize=10, thumb_frac=0.05, cmap='gray_r')

        # ----- embedding:
        if embedding_method == 'LLE':
            clf = LLE(n_neighbors=5, n_components=n_components, method='standard')
            clf.fit(X=X_train)
            X_train_projected = clf.transform(X=X_train)
            X_test_projected = clf.transform(X=X_test)
        elif embedding_method == 'Isomap':
            clf = Isomap(n_neighbors=5, n_components=n_components)
            clf.fit(X=X_train)
            X_train_projected = clf.transform(X=X_train)
            X_test_projected = clf.transform(X=X_test)
        elif embedding_method == 'MDS':
            clf = MDS(n_components=n_components)
            X_projected = clf.fit_transform(X=np.vstack([X_train, X_test]))
            X_train_projected = X_projected[:X_train.shape[0], :]
            X_test_projected = X_projected[X_train.shape[0]:, :]
        elif embedding_method == 'PCA':
            clf = PCA(n_components=n_components)
            clf.fit(X=X_train)
            X_train_projected = clf.transform(X=X_train)
            X_test_projected = clf.transform(X=X_test)
        elif embedding_method == 'KernelPCA':
            clf = KernelPCA(n_components=n_components, kernel='rbf')
            clf.fit(X=X_train)
            X_train_projected = clf.transform(X=X_train)
            X_test_projected = clf.transform(X=X_test)
        elif embedding_method == 'LaplacianEigenmap':
            clf = LaplacianEigenmap(n_neighbors=5, n_components=n_components)
            X_projected = clf.fit_transform(X=np.vstack([X_train, X_test]))
            X_train_projected = X_projected[:X_train.shape[0], :]
            X_test_projected = X_projected[X_train.shape[0]:, :]
        elif embedding_method == 'LDA':
            clf = LDA(n_components=n_components)
            clf.fit(X=X_train, y=y_train)
            X_train_projected = clf.transform(X=X_train)
            X_test_projected = clf.transform(X=X_test)
        elif embedding_method == 'SPCA':
            clf = SPCA(n_components=n_components)
            clf.fit(X=X_train, y=y_train)
            X_train_projected = clf.transform(X=X_train)
            X_test_projected = clf.transform(X=X_test)
        elif embedding_method == 'TSNE':
            clf = TSNE(n_components=min(3, n_components))
            # print(type(list(y_train)))
            X_projected = clf.fit_transform(X=np.vstack([X_train, X_test]),
                                            y=np.asarray(list(y_train) + list(y_test)))
            X_train_projected = X_projected[:X_train.shape[0], :]
            X_test_projected = X_projected[X_train.shape[0]:, :]
        elif embedding_method == 'ML':
            clf = ML(n_components=n_components)
            clf.fit(X=X_train, y=y_train)
            X_train_projected = clf.transform(X=X_train)
            X_test_projected = clf.transform(X=X_test)
        elif embedding_method == 'Kernel_FLDA':
            clf = Kernel_FLDA(n_components=n_components, kernel='linear')
            clf.fit(X=X_train, y=y_train)
            X_train_projected = clf.transform(X=X_train)
            X_test_projected = clf.transform(X=X_test)
        elif embedding_method == 'No_embedding':
            X_train_projected = X_train
            X_test_projected = X_test

        # --- classification:
        print('Classification...')
        # clf = KNN(n_neighbors=1)
        clf = NB()
        clf.fit(X=X_train_projected, y=y_train)
        y_pred = clf.predict(X=X_test_projected)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        error = 1 - accuracy_score(y_true=y_test, y_pred=y_pred)

        # --- saving results:
        save_variable(accuracy, 'accuracy', path_to_save='./output/MNIST/')
        save_np_array_to_txt(np.asarray(accuracy), 'accuracy', path_to_save='./output/MNIST/')
        save_variable(error, 'error', path_to_save='./output/MNIST/')
        save_np_array_to_txt(np.asarray(error), 'error', path_to_save='./output/MNIST/')
        # --- report results:
        print(' ')
        print('Accuracy: ', accuracy * 100)
        print(' ')
        print('Error: ', error * 100)

#----------------------------------------------------------------------
# functions:

# --> good webs for code on plotting manifolds:
# https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
# http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

###### ----------- Read MNIST dataset:

def read_MNIST_dataset(dataset = "training", path = "."):
    # https://gist.github.com/akesling/5358964
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        print('error.....')
    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    get_img = lambda idx: (lbl[idx], img[idx])
    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def show_mnist_data(image):
    # https://gist.github.com/akesling/5358964
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

###### ----------- Read Facial dataset:

def read_image_dataset(dataset_path, imagesType='.png'):
    image_list, image_shape = read_images(folder_path=dataset_path, imagesType=imagesType)
    number_of_images = len(image_list)
    number_of_samples_of_each_class = 10
    X = []; y = []
    for image_index in range(number_of_images):
        class_index = int(image_index / number_of_samples_of_each_class)
        image = image_list[image_index]
        X.append(image)
        y.append(class_index)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, image_shape

def read_images(folder_path='./', imagesType='.png'):
    image_list = []
    images_address = folder_path + '*' + imagesType
    for filename in natsort(list_=glob.glob(images_address)):
        im = Image.open(filename)    # similar to: im = plt.imread(filename)
        image_shape = np.asarray(im).shape
        im = np.asarray(im).ravel()
        image_list.append(im)
    return image_list, image_shape

def natsort(list_):
    """ for sorting names of files in human-sense """
    # http://code.activestate.com/recipes/285264-natural-string-sorting/  ---> comment of r8qyfhp02
    # decorate
    tmp = [ (int(re.search('\d+', i).group(0)), i) for i in list_ ]
    tmp.sort()
    # undecorate
    return [ i[1] for i in tmp ]

###### ----------- Read breast cancer dataset:

def read_BreastCancer_dataset(data, labels_of_classes):
    data = data.values  # converting pandas dataFrame to numpy array
    labels = data[:,1]
    total_number_of_samples = data.shape[0]
    X = data[:,2:]
    X = X.astype(np.float32)  # if we don't do that, we will have this error: https://www.reddit.com/r/learnpython/comments/7ivopz/numpy_getting_error_on_matrix_inverse/
    y = [None] * (total_number_of_samples)  # numeric labels
    for sample_index in range(total_number_of_samples):
        if labels[sample_index] == labels_of_classes[0]:  # first class
            y[sample_index] = 0
        elif labels[sample_index] == labels_of_classes[1]:  # second class
            y[sample_index] = 1
    return X, y

def cross_validation(X, y, n_splits=10, test_size=0.3):
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=None)
    train_indices_in_folds = []; test_indices_in_folds = []
    X_train_in_folds = []; X_test_in_folds = []
    y_train_in_folds = []; y_test_in_folds = []
    for train_index, test_index in sss.split(X, y):
        train_indices_in_folds.append(train_index)
        test_indices_in_folds.append(test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = np.asarray(y)[train_index], np.asarray(y)[test_index]
        X_train_in_folds.append(X_train)
        X_test_in_folds.append(X_test)
        y_train_in_folds.append(y_train)
        y_test_in_folds.append(y_test)
    return train_indices_in_folds, test_indices_in_folds, X_train_in_folds, X_test_in_folds, y_train_in_folds, y_test_in_folds

###### ----------- Save variables:

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def save_np_array_to_txt(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.txt'
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    with open(file_address, 'w') as f:
        f.write(np.array2string(variable, separator=', '))

###### ----------- Plot images in embedded space:

def plot_components(X_projected, images=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    ax = ax or plt.gca()
    ax.plot(X_projected[:, 0], X_projected[:, 1], '.k', markersize=markersize)
    # images = images[:, ::image_scale, ::image_scale]  # downsample the images
    # images = imresize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale))   # downsample the images
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True)
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                                      X_projected[i])
            ax.add_artist(imagebox)
    plt.show()

if __name__ == '__main__':
    main()
