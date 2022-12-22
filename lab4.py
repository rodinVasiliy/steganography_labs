import copy
import os

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lab3 import show_tesing_results


names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=10000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]


def get_end_index(array):
    for i in range(len(array) - 1, -1, -1):
        if array[i] != 0:
            return i
    # по идее такого не будет никогда, но выглядит как костыль
    return 0


def get_series_length(array):
    ravel_array = np.ravel(array)
    max_series = 1000
    series_length_array = np.zeros(max_series)

    current_bit = 0
    series_value = 0
    N = ravel_array.size
    while current_bit < N:
        for series_bit in range(current_bit, N):
            if ravel_array[current_bit] == ravel_array[series_bit]:
                series_value += 1
            else:
                if series_value < max_series:
                    series_length_array[series_value - 1] += 1
                    current_bit += series_value
                    series_value = 0
                    break
            if series_bit == N - 1:
                current_bit = N
    index = get_end_index(series_length_array)
    series_length_array = series_length_array[0:index]
    return series_length_array / N


def add_to_first_bit_plate(C, q):
    C_copy = copy.copy(C)
    N = int(C.size * q)
    W = np.random.randint(0, 2, size=N).astype(np.uint8)
    coords = np.random.permutation(C.size)[:N]
    # зануляем битовую плоскость и добавляем цвз
    bit_num = 1
    num_for_clear_bit_plate = 255 - (2 ** (bit_num - 1))
    C_copy.flat[coords] & num_for_clear_bit_plate
    C_copy.flat[coords] += W
    return C_copy


def get_w(series_length_array, params):
    count = params[0]
    if count > len(series_length_array):
        w = np.zeros(count)
        for i in range(0, len(series_length_array)):
            w[i] = series_length_array[i]
        return w
    w = series_length_array[0:count]
    return w


if __name__ == "__main__":
    images_file_names = os.listdir('resources/images')
    file_names = []
    for file_name in images_file_names:
        file_names.append(os.path.join('resources', 'images', file_name))

    pictures = np.array([plt.imread(file) for file in file_names])
    K = int(len(file_names))
    q_array = [1, 0.7, 0.5, 0.3, 0.1]
    max_count_series = 50
    param = [15]
    accuracy_array = []
    cnt = 0
    for q in q_array:
        print(f'iteration number : {cnt}')
        cnt += 1
        pictures_copy = copy.copy(pictures)
        for i in range(0, int(K / 2)):
            pictures_copy[i] = add_to_first_bit_plate(pictures_copy[i], q)
        data = []
        for i in range(0, K):
            data.append(get_w(get_series_length(pictures_copy[i]), param))
        data = np.array(data)

        target = np.zeros(shape=K)
        target[int(K / 2):K] = 1

        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.8, random_state=0,
                                                            stratify=target)

        name0 = names[5]
        clf0 = classifiers[5]
        clf0.fit(x_train, y_train)
        y_pred0 = clf0.predict(x_test)

        name1 = names[2]
        clf1 = classifiers[2]
        clf1.fit(x_train, y_train)
        y_pred1 = clf1.predict(x_test)

        accuracy_array.append([accuracy_score(y_test, y_pred0), accuracy_score(y_test, y_pred1)])
    accuracy_array = np.array(accuracy_array)
    print(accuracy_array)
    y0 = accuracy_array[:, 0]
    y1 = accuracy_array[:, 1]
    show_tesing_results(q_array, y0, 'q', 'accuracy score', 'first classifier')
    show_tesing_results(q_array, y1, 'q', 'accuracy score', 'second classifier')
    plt.show()