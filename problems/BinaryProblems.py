import copy
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from deap.benchmarks.binary import chuang_f1
from sklearn import datasets, neighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from problems import KnapsackConstants as kpc


class BinaryProblems(object):
    def __init__(self, name, dim, minf, maxf, number_objectives):
        self.name = name
        self.minf = minf
        self.maxf = maxf
        self.number_objectives = number_objectives
        self.dim = dim
        self.is_feature_selection = False

    def fs_evaluate_aux(self, x, filtered_df, selected_features):
        # split data 70% for trainning and 30% to test
        X_train, X_test, y_train, y_test = train_test_split(filtered_df, self.dataset_target, test_size=0.30)

        # 10 fold cross-validation to calculate objective_function
        scores = cross_val_score(self.clf, X_train, y_train, cv=10)
        accuracy = scores.mean()
        sf_n = 1 - float(selected_features) / (len(x))

        # train classifier
        self.clf.fit(X_train, y_train)

        # train and test accuracy results
        y_pred_train = self.clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)

        y_pred_test = self.clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)

        return [(accuracy * self.alpha) + ((1 - self.alpha) * sf_n), test_acc, train_acc, selected_features]

    def evaluate(self, x):
        pass


class OneMax(BinaryProblems):
    def __init__(self, number_variables, number_objectives=1):
        super(OneMax, self).__init__('OneMax', number_variables, 0, 1, number_objectives)

    def evaluate(self, x):
        return [float(sum(x)), 0, 0, 0]


class ZeroMax(BinaryProblems):
    def __init__(self, number_variables, number_objectives=1):
        super(ZeroMax, self).__init__('ZeroMax', number_variables, 0, 1, number_objectives)

    def evaluate(self, val):
        return [np.count_nonzero(np.asarray(val) == 0), 0, 0, 0]


class ChuangF1(BinaryProblems):
    def __init__(self, number_variables, number_objectives=2):
        super(ChuangF1, self).__init__('ChuangF1', number_variables, 0, 1, number_objectives)

    def evaluate(self, val):
        return [float(chuang_f1(val)[0]), 0, 0, 0]


class ZeroOneKnapsack(BinaryProblems):
    def __init__(self, dim):
        self.dim = dim
        dim_values = {10: [kpc.P_10, kpc.W_10, kpc.C_10],
                      15: [kpc.P_15, kpc.W_15, kpc.C_15],
                      20: [kpc.P_20, kpc.W_20, kpc.C_20],
                      30: [kpc.P_30, kpc.W_30, kpc.C_30],
                      50: [kpc.P_50, kpc.W_50, kpc.C_50],
                      100: [kpc.P_100, kpc.W_100, kpc.C_100],
                      200: [kpc.P_200, kpc.W_200, kpc.C_200],
                      300: [kpc.P_300, kpc.W_300, kpc.C_300],
                      400: [kpc.P_400, kpc.W_400, kpc.C_400],
                      500: [kpc.P_500, kpc.W_500, kpc.C_500],
                      600: [kpc.P_600, kpc.W_600, kpc.C_600],
                      700: [kpc.P_700, kpc.W_700, kpc.C_700],
                      800: [kpc.P_800, kpc.W_800, kpc.C_800],
                      900: [kpc.P_900, kpc.W_900, kpc.C_900],
                      1000: [kpc.P_1000, kpc.W_1000, kpc.C_1000]}

        if dim not in [10, 15, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            raise Exception("Number of dimensions must be 10, 15, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900"
                            " or 1000")

        self.profit, self.weight, self.capacity = dim_values[self.dim]
        super(ZeroOneKnapsack, self).__init__('Knapsack', dim, 0, 1, None)

    def evaluate(self, positions):
        sum_profit = 0.0
        sum_weight = 0.0
        for i in range(self.dim):
            sum_profit += self.profit[i] * positions[i]
            sum_weight += self.weight[i] * positions[i]
        if sum_weight <= self.capacity:
            return [sum_profit, 0, 0, 0]
        else:
            return [0.0, 0, 0, 0]


# In this implementation each class will have (dim/num_class) elements
class MultipleChoiceKnapsack(BinaryProblems):
    def __init__(self, dim, num_class):
        self.dim = dim
        self.num_class = num_class
        dim_values = {10: [kpc.P_10, kpc.W_10, kpc.C_10],
                      15: [kpc.P_15, kpc.W_15, kpc.C_15],
                      20: [kpc.P_20, kpc.W_20, kpc.C_20],
                      30: [kpc.P_30, kpc.W_30, kpc.C_30],
                      50: [kpc.P_50, kpc.W_50, kpc.C_50],
                      100: [kpc.P_100, kpc.W_100, kpc.C_100],
                      200: [kpc.P_200, kpc.W_200, kpc.C_200],
                      300: [kpc.P_300, kpc.W_300, kpc.C_300],
                      400: [kpc.P_400, kpc.W_400, kpc.C_400],
                      500: [kpc.P_500, kpc.W_500, kpc.C_500],
                      600: [kpc.P_600, kpc.W_600, kpc.C_600],
                      700: [kpc.P_700, kpc.W_700, kpc.C_700],
                      800: [kpc.P_800, kpc.W_800, kpc.C_800],
                      900: [kpc.P_900, kpc.W_900, kpc.C_900],
                      1000: [kpc.P_1000, kpc.W_1000, kpc.C_1000]}

        if dim not in [10, 15, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            raise Exception("Number of dimensions must be 10, 15, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900"
                            " or 1000")
        if dim % num_class != 0:
            raise Exception("Number of dimensions must be a multiple of number of classes")

        self.profit, self.weight, self.capacity = dim_values[self.dim]
        super(MultipleChoiceKnapsack, self).__init__('Multiple Choice Knapsack', dim, 0, 1, None)

    def evaluate(self, positions):
        sum_profit = 0.0
        sum_weight = 0.0
        pos_class = np.array_split(positions, self.num_class)

        if sum([sum(x) for x in pos_class if sum(x) == 1]) != self.num_class:
            # print (positions, "Zero")
            return [0.0, 0, 0, 0]
        else:
            dim = 0
            for c in pos_class:
                for i in range(len(c)):
                    sum_profit += self.profit[dim] * positions[i]
                    sum_weight += self.weight[dim] * positions[i]
                    dim += 1
            if sum_weight <= self.capacity:
                return [sum_profit, 0, 0, 0]
            else:
                return [0.0, 0, 0, 0]


class ChangeMaking(BinaryProblems):
    def __init__(self, dim):
        self.dim = dim
        dim_values = {10: [kpc.W_10, kpc.C_10],
                      15: [kpc.W_15, kpc.C_15],
                      20: [kpc.W_20, kpc.C_20],
                      30: [kpc.W_30, kpc.C_30],
                      50: [kpc.W_50, kpc.C_50],
                      100: [kpc.W_100, kpc.C_100],
                      200: [kpc.W_200, kpc.C_200],
                      300: [kpc.W_300, kpc.C_300],
                      400: [kpc.W_400, kpc.C_400],
                      500: [kpc.W_500, kpc.C_500],
                      600: [kpc.W_600, kpc.C_600],
                      700: [kpc.W_700, kpc.C_700],
                      800: [kpc.W_800, kpc.C_800],
                      900: [kpc.W_900, kpc.C_900],
                      1000: [kpc.W_1000, kpc.C_1000]}

        if dim not in [10, 15, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            raise Exception("Number of dimensions must be 10, 15, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900"
                            " or 1000")

        self.weight, self.change = dim_values[self.dim]
        super(ChangeMaking, self).__init__('Change Making', dim, 0, 1, None)

    def evaluate(self, positions):
        sum_pos = 0.0
        sum_weight = 0.0
        for i in range(self.dim):
            sum_pos += positions[i]
            sum_weight += self.weight[i] * positions[i]
        if sum_weight == self.change:
            return [-sum_pos, 0, 0, 0]
        else:
            return [-10 ** 5, 0, 0, 0]


class SubsetSum(BinaryProblems):
    def __init__(self, dim):
        self.dim = dim
        dim_values = {10: [kpc.W_10, kpc.C_10],
                      15: [kpc.W_15, kpc.C_15],
                      20: [kpc.W_20, kpc.C_20],
                      30: [kpc.W_30, kpc.C_30],
                      50: [kpc.W_50, kpc.C_50],
                      100: [kpc.W_100, kpc.C_100],
                      200: [kpc.W_200, kpc.C_200],
                      300: [kpc.W_300, kpc.C_300],
                      400: [kpc.W_400, kpc.C_400],
                      500: [kpc.W_500, kpc.C_500],
                      600: [kpc.W_600, kpc.C_600],
                      700: [kpc.W_700, kpc.C_700],
                      800: [kpc.W_800, kpc.C_800],
                      900: [kpc.W_900, kpc.C_900],
                      1000: [kpc.W_1000, kpc.C_1000]}

        if dim not in [10, 15, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            raise Exception("Number of dimensions must be 10, 15, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900"
                            " or 1000")

        self.weight, self.capacity = dim_values[self.dim]
        super(SubsetSum, self).__init__('Subset Sum', dim, 0, 1, None)

    def evaluate(self, positions):
        sum_weight = 0.0
        for i in range(self.dim):
            sum_weight += self.weight[i] * positions[i]
        if sum_weight <= self.capacity:
            return [sum_weight, 0, 0, 0]
        else:
            return [0.0, 0, 0, 0]


# In this implementation each class will have (dim/num_knapsacks) dimensions with the same capacity
class MultipleKnapsack(BinaryProblems):
    def __init__(self, dim, num_knapsacks):
        self.dim = dim
        self.num_knapsacks = num_knapsacks
        dim_values = {10: [kpc.P_10, kpc.W_10, kpc.C_10],
                      15: [kpc.P_15, kpc.W_15, kpc.C_15],
                      20: [kpc.P_20, kpc.W_20, kpc.C_20],
                      30: [kpc.P_30, kpc.W_30, kpc.C_30],
                      50: [kpc.P_50, kpc.W_50, kpc.C_50],
                      100: [kpc.P_100, kpc.W_100, kpc.C_100],
                      200: [kpc.P_200, kpc.W_200, kpc.C_200],
                      300: [kpc.P_300, kpc.W_300, kpc.C_300],
                      400: [kpc.P_400, kpc.W_400, kpc.C_400],
                      500: [kpc.P_500, kpc.W_500, kpc.C_500],
                      600: [kpc.P_600, kpc.W_600, kpc.C_600],
                      700: [kpc.P_700, kpc.W_700, kpc.C_700],
                      800: [kpc.P_800, kpc.W_800, kpc.C_800],
                      900: [kpc.P_900, kpc.W_900, kpc.C_900],
                      1000: [kpc.P_1000, kpc.W_1000, kpc.C_1000]}

        if dim not in [10, 15, 20, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            raise Exception("Number of dimensions must be 10, 15, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900"
                            " or 1000")
        if dim % num_knapsacks != 0:
            raise Exception("Number of dimensions must be a multiple of number of knapsack")

        self.profit, self.weight, self.capacity = dim_values[self.dim]
        super(MultipleKnapsack, self).__init__('Multiple Knapsack', dim, 0, 1, None)

    def evaluate(self, positions):
        sum_profit = 0.0
        knapsacks = np.array_split(positions, self.num_knapsacks)
        dim = 0
        for knapsack in knapsacks:
            sum_weight = 0
            for i in range(len(knapsack)):
                sum_profit += self.profit[dim] * knapsack[i]
                sum_weight += self.weight[dim] * knapsack[i]
                dim += 1
            if sum_weight > self.capacity:
                return [0.0, 0, 0, 0]
        else:
            return [sum_profit, 0, 0, 0]


class MultidimensionalKnapsack(BinaryProblems):
    def __init__(self, dim):
        self.dim = dim
        dim_values = {10: [kpc.P_10, kpc.W_10, kpc.V_10, kpc.C_10, kpc.CV_10],
                      15: [kpc.P_15, kpc.W_15, kpc.V_15, kpc.C_15, kpc.CV_15],
                      20: [kpc.P_20, kpc.W_20, kpc.V_20, kpc.C_20, kpc.CV_20],
                      30: [kpc.P_30, kpc.W_30, kpc.V_30, kpc.C_30, kpc.CV_30],
                      50: [kpc.P_50, kpc.W_50, kpc.V_50, kpc.C_50, kpc.CV_50],
                      100: [kpc.P_100, kpc.W_100, kpc.V_100, kpc.C_100, kpc.CV_100],
                      500: [kpc.P_500, kpc.W_500, kpc.V_500, kpc.C_500, kpc.CV_500],
                      1000: [kpc.P_1000, kpc.W_1000, kpc.V_1000, kpc.C_1000, kpc.CV_1000]}

        if dim not in [10, 15, 20, 30, 50, 100, 500, 1000]:
            raise Exception("Number of dimensions must be 10, 15, 30, 50, 100, 500 or 1000")

        self.profit, self.weight, self.volume, self.capacity, self.v_capacity = dim_values[self.dim]
        super(MultidimensionalKnapsack, self).__init__('Multidimensional Knapsack', dim, 0, 1, None)

    def evaluate(self, positions):
        sum_profit = 0.0
        sum_weight = 0.0
        sum_volume = 0.0
        for i in range(self.dim):
            sum_profit += self.profit[i] * positions[i]
            sum_weight += self.weight[i] * positions[i]
            sum_volume += self.volume[i] * positions[i]
        if sum_weight <= self.capacity and sum_volume <= self.v_capacity:
            return [sum_profit, 0, 0, 0]
        else:
            return [0.0, 0, 0, 0]


class FSIsolet5(BinaryProblems):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha
        self.clf = None
        super(FSIsolet5, self).__init__('Feature Selection Isolet 5', dim, 0, 1, None)
        self.is_feature_selection = True

        files_loc = 'datasets/isolet5'
        data = pd.read_csv(os.path.join(files_loc, 'isolet5.data'), delimiter=',', header=None)

        # Separating data from labels
        lb = LabelEncoder()
        self.dataset_data = data[range(self.dim)]
        self.dataset_target = lb.fit_transform(np.ravel(data[[self.dim]], order='C'))

    def evaluate(self, x):
        excluded = np.where(x == 0)[0]
        selected_features = len(np.where(x == 1)[0])
        filtered_df = copy.deepcopy(self.dataset_data)

        if selected_features > 0:
            for d in excluded:
                filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSMadelon(BinaryProblems):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha
        self.clf = None
        super(FSMadelon, self).__init__('Feature Selection Madelon', dim, 0, 1, None)
        self.is_feature_selection = True

        files_loc = 'datasets/madelon'
        x_train = pd.read_csv(os.path.join(files_loc, 'madelon_train.data'), delimiter=' ', header=None)
        y_train = pd.read_csv(os.path.join(files_loc, 'madelon_train.labels'), delimiter=' ', header=None)

        x_test = pd.read_csv(os.path.join(files_loc, 'madelon_valid.data'), delimiter=' ', header=None)
        y_test = pd.read_csv(os.path.join(files_loc, 'madelon_valid.labels'), delimiter=' ', header=None)

        # removing empty columns
        x_train.drop([500], 1, inplace=True)
        x_test.drop([500], 1, inplace=True)

        # Concatenate datasets
        self.dataset_data = [x_train, x_test]
        temp = pd.concat([y_train, y_test])
        lb = LabelEncoder()
        self.dataset_target = lb.fit_transform(np.ravel(temp, order='C'))

    def evaluate(self, x):
        excluded = np.where(x == 0)[0]
        filtered_df = pd.concat(self.dataset_data)
        selected_features = len(np.where(x == 1)[0])

        if selected_features > 0:
            for d in excluded:
                filtered_df.drop([d], 1, inplace=True)

            # create classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSMusk1(BinaryProblems):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha
        self.clf = None
        super(FSMusk1, self).__init__('Feature Selection Musk 1', dim, 0, 1, None)
        self.is_feature_selection = True

        files_loc = 'datasets/musk1'
        data = pd.read_csv(os.path.join(files_loc, 'musk1.data'), delimiter=',', header=None)

        # Removing class name columns
        data.drop([0], 1, inplace=True)
        data.drop([1], 1, inplace=True)
        data.columns = range(self.dim + 1)

        # Separating data from labels
        self.dataset_data = data[range(self.dim)]
        lb = LabelEncoder()
        self.dataset_target = lb.fit_transform(np.ravel(data[[self.dim]], order='C'))

    def evaluate(self, x):
        excluded = np.where(x == 0)[0]
        filtered_df = copy.deepcopy(self.dataset_data)
        selected_features = len(np.where(x == 1)[0])

        if selected_features > 0:
            for d in excluded:
                filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSHillValley(BinaryProblems):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha
        self.clf = None
        super(FSHillValley, self).__init__('Feature Selection Hill-Valley', dim, 0, 1, None)
        self.is_feature_selection = True

        files_loc = 'datasets/hill-valley'
        data = pd.read_csv(os.path.join(files_loc, 'hill-valley.data'), delimiter=',', header=0)

        # replacing columns names
        data.columns = range(self.dim + 1)

        # Separating data from labels
        self.dataset_data = data[range(self.dim)]
        lb = LabelEncoder()
        self.dataset_target = lb.fit_transform(np.ravel(data[[self.dim]], order='C'))

    def evaluate(self, x):
        excluded = np.where(x == 0)[0]
        filtered_df = copy.deepcopy(self.dataset_data)
        selected_features = len(np.where(x == 1)[0])

        if selected_features > 0:
            for d in excluded:
                filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSLung(BinaryProblems):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha
        self.clf = None
        super(FSLung, self).__init__('Feature Selection Lung Cancer', dim, 0, 1, None)
        self.is_feature_selection = True

        files_loc = 'datasets/lung'
        data = pd.read_csv(os.path.join(files_loc, 'lung.data'), delimiter=',', header=None)

        # Separating data from labels
        self.dataset_data = data[range(self.dim)]
        lb = LabelEncoder()
        self.dataset_target = lb.fit_transform(np.ravel(data[[self.dim]], order='C'))

    def evaluate(self, x):
        excluded = np.where(x == 0)[0]
        filtered_df = copy.deepcopy(self.dataset_data)
        selected_features = len(np.where(x == 1)[0])

        if selected_features > 0:
            for d in excluded:
                filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSIonosphere(BinaryProblems):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha
        self.clf = None
        super(FSIonosphere, self).__init__('Feature Selection Ionosphere', dim, 0, 1, None)
        self.is_feature_selection = True

        files_loc = 'datasets/ionosphere'
        data = pd.read_csv(os.path.join(files_loc, 'ionosphere.data'), delimiter=',', header=None)

        # Separating data from labels
        self.dataset_data = data[range(self.dim)]
        lb = LabelEncoder()
        self.dataset_target = lb.fit_transform(np.ravel(data[[self.dim]], order='C'))

    def evaluate(self, x):
        excluded = np.where(x == 0)[0]
        filtered_df = copy.deepcopy(self.dataset_data)
        selected_features = len(np.where(x == 1)[0])

        if selected_features > 0:
            for d in excluded:
                filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSWDBC(BinaryProblems):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha
        self.clf = None
        super(FSWDBC, self).__init__('Feature Selection WDBC', dim, 0, 1, None)
        self.is_feature_selection = True

        files_loc = 'datasets/wdbc'
        data = pd.read_csv(os.path.join(files_loc, 'wdbc.data'), delimiter=',', header=None)
        data.drop([0], 1, inplace=True)
        data.columns = range(self.dim + 1)

        # Separating data from labels
        self.dataset_data = data[range(1, self.dim + 1)]
        self.dataset_data.columns = range(self.dim)
        lb = LabelEncoder()
        self.dataset_target = lb.fit_transform(np.ravel(data[[0]], order='C'))

    def evaluate(self, x):
        excluded = np.where(x == 0)[0]
        filtered_df = copy.deepcopy(self.dataset_data)
        selected_features = len(np.where(x == 1)[0])

        if selected_features > 0:
            for d in excluded:
                filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSGerman(BinaryProblems):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha
        self.clf = None
        super(FSGerman, self).__init__('Feature Selection German', dim, 0, 1, None)
        self.is_feature_selection = True

        files_loc = 'datasets/german'
        data = pd.read_csv(os.path.join(files_loc, 'german.data'), delimiter=' ', header=None)
        data.drop([0], 1, inplace=True)
        data.drop([26], 1, inplace=True)
        data.columns = range(self.dim + 1)

        # Separating data from labels
        self.dataset_data = data[range(self.dim)]
        self.dataset_data.columns = range(self.dim)
        lb = LabelEncoder()
        self.dataset_target = lb.fit_transform(np.ravel(data[self.dim], order='C'))

    def evaluate(self, x):
        excluded = np.where(x == 0)[0]
        filtered_df = copy.deepcopy(self.dataset_data)
        selected_features = len(np.where(x == 1)[0])

        if selected_features > 0:
            for d in excluded:
                filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSVehicle(BinaryProblems):
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha
        self.clf = None
        super(FSVehicle, self).__init__('Feature Selection Vehicle', dim, 0, 1, None)
        self.is_feature_selection = True

        files_loc = 'datasets/vehicle'
        data = pd.read_csv(os.path.join(files_loc, 'vehicle.data'), delimiter=' ', header=None)
        data.drop([19], 1, inplace=True)

        # Separating data from labels
        self.dataset_data = data[range(self.dim)]
        self.dataset_data.columns = range(self.dim)
        lb = LabelEncoder()
        self.dataset_target = lb.fit_transform(np.ravel(data[[self.dim]], order='C'))

    def evaluate(self, x):
        excluded = np.where(x == 0)[0]
        filtered_df = copy.deepcopy(self.dataset_data)
        selected_features = len(np.where(x == 1)[0])

        if selected_features > 0:
            for d in excluded:
                filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSWine(BinaryProblems):
    def __init__(self, dim, alpha):
        self.clf = None
        self.dim = dim
        self.alpha = alpha
        super(FSWine, self).__init__('Feature Selection Wine', dim, 0, 1, None)
        self.is_feature_selection = True

        # Concatenate datasets
        data = datasets.load_wine()
        self.dataset_data, self.dataset_target = data.data, data.target

    def evaluate(self, x):
        selected_features = len(np.where(x == 1)[0])
        if selected_features > 0:
            excluded = np.where(x == 0)[0]
            filtered_df = copy.deepcopy(self.dataset_data)

            # for d in excluded:
            filtered_df = np.delete(filtered_df, excluded, 1)
            # filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSGene(BinaryProblems):
    def __init__(self, dim, alpha):
        self.clf = None
        self.dim = dim
        self.alpha = alpha
        super(FSGene, self).__init__('Feature Selection TGCA PANCAN', dim, 0, 1, None)
        self.is_feature_selection = True

        lb = LabelEncoder()
        data = pd.read_csv(os.path.join('datasets', 'tcga', 'tcga_pancan_data.csv'), delimiter=',', header='infer')
        data = data.drop(columns="Unnamed: 0")
        targets = pd.read_csv(os.path.join('datasets', 'tcga', 'tcga_pancan_labels.csv'), delimiter=',', header='infer')
        targets = lb.fit_transform(np.ravel(targets.drop(columns="Unnamed: 0"), order='C'))
        self.dataset_data, self.dataset_target = data.values, targets

    def evaluate(self, x):
        selected_features = len(np.where(x == 1)[0])
        if selected_features > 0:
            excluded = np.where(x == 0)[0]
            filtered_df = copy.deepcopy(self.dataset_data)

            # for d in excluded:
            filtered_df = np.delete(filtered_df, excluded, 1)
            # filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness


class FSParkinson(BinaryProblems):
    def __init__(self, dim, alpha):
        self.clf = None
        self.dim = dim
        self.alpha = alpha
        super(FSParkinson, self).__init__('Feature Selection Parkinson', dim, 0, 1, None)
        self.is_feature_selection = True

        lb = LabelEncoder()
        data = pd.read_csv(os.path.join('datasets', 'parkinson', 'pd_speech_features.csv'), delimiter=',',
                           header='infer').drop(columns="id")
        targets = data.pop('class')
        self.dataset_data, self.dataset_target = data.values, targets

    def evaluate(self, x):
        selected_features = len(np.where(x == 1)[0])
        if selected_features > 0:
            excluded = np.where(x == 0)[0]
            filtered_df = copy.deepcopy(self.dataset_data)

            # for d in excluded:
            filtered_df = np.delete(filtered_df, excluded, 1)
            # filtered_df.drop([d], 1, inplace=True)

            # create the classifier
            # self.clf = svm.SVC(kernel='rbf', gamma='auto', C=100)
            self.clf = neighbors.KNeighborsClassifier(n_neighbors=5)

            # normalize data
            std = MinMaxScaler()
            filtered_df = std.fit_transform(filtered_df)

            fitness = self.fs_evaluate_aux(x, filtered_df, selected_features)

        else:
            fitness = [0, 0, 0, 0]

        return fitness
