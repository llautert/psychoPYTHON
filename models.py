from sklearn import *
import matplotlib as mpl

mpl.use('Agg')
from utils import plot_learning_curve
from keras.losses import *
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from multiprocessing import Pool


def deep_1net(X, y):
    # split data
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)

    # transform to_categorical data
    Y_test = to_categorical(Y_test)
    Y_train = to_categorical(Y_train)

    model = Sequential()
    model.add(Dense(units=3, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.compile(loss=mean_absolute_error,
                  optimizer=SGD(lr=0.01),
                  metrics=['accuracy', 'mae'])
    model.fit(X_train, Y_train, epochs=10, batch_size=30, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=1)
    model.save("deep1net.h5")
    print('\nTest loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test mae:', score[2])


def select_models(X_train, type_pred, is_labeled_data, is_text_data, is_number_categories_known,
                  is_few_important_features, is_just_looking):
    names = []
    models = []
    if type_pred == 'category':
        if is_labeled_data:
            # classification
            names.append("SVM")
            models.append(svm.SVC())
            if is_text_data:
                names += ["GaussianNB",
                          "MultinomialNB",
                          "BernoulliNB"]
                models += [naive_bayes.GaussianNB(),
                           naive_bayes.MultinomialNB(),
                           naive_bayes.BernoulliNB()]
            else:
                names += ["KNeighborsClassifier",
                          "LinearSVMClassifier",
                          "AdaboostClassifier",
                          "BaggingClassifier",
                          "ExtraTreesClassifier",
                          "GradientBoostingClassifier",
                          "RandomForestClassifier",
                          "SGDClassifier",
                          "AdditiveChi2Sampler",
                          "Nystroem",
                          "RBFSampler",
                          "SkewedChi2Sampler"]
                models += [neighbors.KNeighborsClassifier(),
                           svm.LinearSVC(max_iter=10),
                           ensemble.AdaBoostClassifier(),
                           ensemble.BaggingClassifier(),
                           ensemble.ExtraTreesClassifier(),
                           ensemble.GradientBoostingClassifier(),
                           ensemble.RandomForestClassifier(),
                           linear_model.SGDClassifier(),
                           kernel_approximation.AdditiveChi2Sampler(),
                           kernel_approximation.Nystroem(),
                           kernel_approximation.RBFSampler(),
                           kernel_approximation.SkewedChi2Sampler()]
        elif is_number_categories_known:
            # clustering
            names += ["KMeans",
                      "MiniBatchKMeans",
                      "GMM"]
            models += [
                cluster.KMeans(),
                mixture.GMM(),
                cluster.MiniBatchKMeans()
            ]
        else:
            names += ["MeanShift",
                      "VBGMM"]
            models += [
                cluster.MeanShift(),
                mixture.VBGMM()
            ]
    elif type_pred == "quantity":
        # regression
        # names.append("SGDRegressor")
        # models.append(linear_model.SGDRegressor())
        if is_few_important_features:
            names += ["Lasso",
                      "ElasticNet"]
            models += [linear_model.Lasso(),
                       linear_model.ElasticNet()]
        else:
            names += ["Ridge",
                      "LinearSVMRegressor",
                      "RBFSVMRegressor",
                      "AdaboostRegressor",
                      "BaggingRegressor",
                      "ExtraTreesRegressor",
                      "GradientBoostingRegressor",
                      "RandomForestRegressor"]
            models += [
                linear_model.Ridge(),
                svm.LinearSVR(max_iter=10),
                svm.SVR(kernel='rbf', max_iter=10),
                ensemble.AdaBoostRegressor(n_estimators=10),
                ensemble.BaggingRegressor(n_jobs=-1),
                ensemble.ExtraTreesRegressor(n_jobs=-1),
                ensemble.GradientBoostingRegressor(n_estimators=10),
                ensemble.RandomForestRegressor(n_jobs=-1)
            ]
    elif is_just_looking:
        # dimensional reduction
        names.append("RandomizedPCA")
        models.append(decomposition.RandomizedPCA())
        names += ["Isomap",
                  "SpectalEmbedding",
                  "LocallyLinearEmbedding",
                  "AdditiveChi2Sampler",
                  "Nystroem",
                  "RBFSampler",
                  "SkewedChi2Sampler"]
        models += [
            manifold.Isomap(),
            manifold.SpectalEmbedding(),
            manifold.LocallyLinearEmbedding(),
            kernel_approximation.AdditiveChi2Sampler(),
            kernel_approximation.Nystroem(),
            kernel_approximation.RBFSampler(),
            kernel_approximation.SkewedChi2Sampler()]
    else:
        print("tough luck")

    return names, models


def train_test_model(args):
    name, m1, X, y, X_train, Y_train, X_test, Y_test, type_pred, is_number_categories_known, is_labeled_data = args
    print(m1)
    if not is_number_categories_known:
        plot_learning_curve(m1, X, y, name, ylim=(0.7, 1.01))
    # with parallel_backend('distributed', scheduler_host='localhost:8786', scatter=[X_train, Y_train]):
    m1.fit(X_train, Y_train)
    externals.joblib.dump(m1, "%s.pkl" % name, compress=9)

    y_pred = m1.predict(X_test)
    results = open('out_%s' % (name), 'w')

    if type_pred == 'category' and is_labeled_data:
        # classification metrics
        results.write("Accuracy Score: %.2f\n" % (metrics.accuracy_score(Y_test, y_pred)))
        results.write("F1 Score: %.2f\n" % (metrics.f1_score(Y_test, y_pred, average="weighted")))
        results.write("Precision Score: %.2f\n" % (metrics.precision_score(Y_test, y_pred, average="weighted")))
        results.write("Recall Score: %.2f\n" % (metrics.recall_score(Y_test, y_pred, average="weighted")))
    elif type_pred == 'category' and is_number_categories_known:
        # clusterization
        results.write("Completeness Score: %.2f\n" % (metrics.completeness_score(Y_test, y_pred)))
        results.write("Homogeneity Score: %.2f\n" % (metrics.homogeneity_score(Y_test, y_pred)))
        results.write("V-Measure Score: %.2f\n" % (metrics.v_measure_score(Y_test, y_pred)))
        results.write(
            "Adjusted Mutual Information Score: %.2f\n" % (metrics.adjusted_mutual_info_score(Y_test, y_pred)))
        results.write("Fowlkes-Mallows index (FMI): %.2f\n" % (metrics.fowlkes_mallows_score(Y_test, y_pred)))
    elif type_pred == 'quantity':
        # regression
        results.write("R2 Score: %.2f\n" % (metrics.r2_score(Y_test, y_pred)))
        results.write("Explained Variance Score: %.2f\n" % (metrics.explained_variance_score(Y_test, y_pred)))
        results.write("Mean Absolute Error: %.2f\n" % (metrics.mean_absolute_error(Y_test, y_pred)))
        results.write("Mean Squared Error: %.2f\n" % (metrics.mean_squared_error(Y_test, y_pred)))
        results.write("Median Absolute Error: %.2f\n" % (metrics.median_absolute_error(Y_test, y_pred)))
    results.close()


def autoscikit(X, y, type_pred='category', is_labeled_data=False, is_text_data=False, is_number_categories_known=False,
               is_few_important_features=False, is_just_looking=False):
    models = []
    names = []
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)

    if X_train.shape[0] > 50:
        names, models = select_models(X_train, type_pred, is_labeled_data, is_text_data, is_number_categories_known,
                                      is_few_important_features, is_just_looking)
        pool = Pool(processes=4)
        # paralelizando modelos
        # register_parallel_backend('distributed', DistributedBackend)
        sequence = []
        # cross validation
        externals.joblib.dump((X, y, X_train, Y_train, X_test, Y_test), "dataset.pkl", compress=9)
        for name, m1 in zip(names, models):
            sequence.append([name, m1, X, y, X_train, Y_train, X_test, Y_test, type_pred, is_number_categories_known,
                             is_labeled_data])
        pool.map(train_test_model, sequence)
        pool.close()
        pool.join()
    else:
        print("Error few data")