from time import time
from sklearn import *
from utils import plot_learning_curve
from keras.losses import *
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

def deep_1net(X, y):
    #split data
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)

    #transform to_categorical data
    Y_test = to_categorical(Y_test)
    Y_train = to_categorical(Y_train)

    model = Sequential()
    model.add(Dense(units=3, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=0.01),
                  metrics=['accuracy', 'mae'])

    model.fit(X_train, Y_train, epochs=10, batch_size=30, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=1)
    model.save("deep1net.h5")
    print('\nTest loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test mae:', score[2])

def choose_models(X_train, type_pred, is_labeled_data, is_text_data, is_number_categories_known, is_few_important_features, is_just_looking):
    names = []
    models = []

    if X_train.shape[0] <= 50:
        print("Error few data")
        return

    if type_pred == 'category':
        if is_labeled_data:
            # classification
            if X_train.shape[0] < 100000:
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
                              "RandomForestClassifier"]
                    models += [neighbors.KNeighborsClassifier(),
                               svm.LinearSVC(),
                               ensemble.AdaBoostClassifier(),
                               ensemble.BaggingClassifier(),
                               ensemble.ExtraTreesClassifier(),
                               ensemble.GradientBoostingClassifier(),
                               ensemble.RandomForestClassifier()]
            else:
                names += ["SGDClassifier",
                          "AdditiveChi2Sampler",
                          "Nystroem",
                          "RBFSampler",
                          "SkewedChi2Sampler"]
                models += [linear_model.SGDClassifier(),
                           kernel_approximation.AdditiveChi2Sampler(),
                           kernel_approximation.Nystroem(),
                           kernel_approximation.RBFSampler(),
                           kernel_approximation.SkewedChi2Sampler()]
        elif is_number_categories_known:
            # clustering
            if X_train.shape[0] < 10000:
                names += ["KMeans",
                          "GMM"]
                models += [cluster.KMeans(),
                           mixture.GMM()]
            else:
                names += ["KMeans",
                          "MiniBatchKMeans"]
                models += [cluster.KMeans(),
                           cluster.MiniBatchKMeans()]
        else:
            if X_train.shape[0] < 10000:
                names += ["MeanShift",
                          "VBGMM"]
                models += [cluster.MeanShift(),
                           mixture.VBGMM()]
            else:
                print("tough luck")
    elif type_pred == "quantity":
        # regression
        if X_train.shape[0] < 1000000000:
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
                models += [linear_model.Ridge(),
                           svm.LinearSVR(),
                           svm.SVR(kernel='rbf'),
                           ensemble.AdaBoostRegressor(),
                           ensemble.BaggingRegressor(),
                           ensemble.ExtraTreesRegressor(),
                           ensemble.GradientBoostingRegressor(),
                           ensemble.RandomForestRegressor()]
        else:
            names.append("SGDRegressor")
            models.append(linear_model.SGDRegressor())
    elif is_just_looking:
        # dimensional reduction
        names.append("RandomizedPCA")
        models.append(decomposition.RandomizedPCA())
        if X_train.shape[0] < 10000:
            names += ["Isomap",
                      "SpectalEmbedding",
                      "LocallyLinearEmbedding"]
            models += [manifold.Isomap(),
                       manifold.SpectalEmbedding(),
                       manifold.LocallyLinearEmbedding()]
        else:
            names += ["AdditiveChi2Sampler",
                      "Nystroem",
                      "RBFSampler",
                      "SkewedChi2Sampler"]
            models += [kernel_approximation.AdditiveChi2Sampler(),
                       kernel_approximation.Nystroem(),
                       kernel_approximation.RBFSampler(),
                       kernel_approximation.SkewedChi2Sampler()]
    else:
        print("tough luck")

    return names, models

def autoscikit(X, y, type_pred='category', is_labeled_data=False, is_text_data=False, is_number_categories_known=False, is_few_important_features=False, is_just_looking=False):
    start_time = time()
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)
    names_and_models = choose_models(X_train, type_pred, is_labeled_data, is_text_data, is_number_categories_known, is_few_important_features, is_just_looking)

    if names_and_models:
        names, models = names_and_models
        # cross validation
        for name, m1 in zip(names, models):
            model_time = time()
            print('\n-----------------------------------------------------------------------------')
            print(m1)
            print('-----------------------------------------------------------------------------')
            if not is_number_categories_known:
                plot_learning_curve(m1, X, y, name, ylim=(0.7, 1.01))
            m1.fit(X_train, Y_train)
            externals.joblib.dump(m1, "%s.pkl" % name, compress=9)

            y_pred = m1.predict(X_test)

            if type_pred == 'category' and is_labeled_data:
                #classification metrics
                print("Accuracy Score: %.2f" % (metrics.accuracy_score(Y_test, y_pred)))
                print("F1 Score: %.2f" % (metrics.f1_score(Y_test, y_pred, average="weighted")))
                print("Precision Score: %.2f" % (metrics.precision_score(Y_test, y_pred, average="weighted")))
                print("Recall Score: %.2f" % (metrics.recall_score(Y_test, y_pred, average="weighted")))
            elif type_pred == 'category' and is_number_categories_known:
                #clusterization
                print("Completeness Score: %.2f" % (metrics.completeness_score(Y_test, y_pred)))
                print("Homogeneity Score: %.2f" % (metrics.homogeneity_score(Y_test, y_pred)))
                print("V-Measure Score: %.2f" % (metrics.v_measure_score(Y_test, y_pred)))
                print("Adjusted Mutual Information Score: %.2f" % (metrics.adjusted_mutual_info_score(Y_test, y_pred)))
                print("Fowlkes-Mallows index (FMI): %.2f" % (metrics.fowlkes_mallows_score(Y_test, y_pred)))
            elif type_pred == 'quantity':
                #regression
                print("R2 Score: %.2f" % (metrics.r2_score(Y_test, y_pred)))
                print("Explained Variance Score: %.2f" % (metrics.explained_variance_score(Y_test, y_pred)))
                print("Mean Absolute Error: %.2f" % (metrics.mean_absolute_error(Y_test, y_pred)))
                print("Mean Squared Error: %.2f" % (metrics.mean_squared_error(Y_test, y_pred)))
                print("Median Absolute Error: %.2f" % (metrics.median_absolute_error(Y_test, y_pred)))

            elapsed_time = time() - model_time
            print("Elapsed time: %.2f seconds" % elapsed_time)

        print('\n-----------------------------------------------------------------------------')
        end_time = time() - start_time
        print("Total elapsed time: %.2f seconds" % end_time)
