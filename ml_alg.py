from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from sklearn.ensemble import GradientBoostingClassifier


def linear_regression(x_train, y_train, normalize=False, x_test = False):
    '''
    linear.coef_, linear.residues_, linear.intercept_
    Options: polynomial regression, different data transformation, interaction.
    :param x_train:
    :param y_train:
    :param normalize:
    :param x_test:
    :return:
    '''
    # Identify feature and response variable(s) and values must be numeric and numpy arrays

    # Create linear regression object
    model = linear_model.LinearRegression(normalize = normalize)

    # Train the model using the training sets and check score
    model.fit(x_train, y_train)
    if x_test == False:
        print('Score:', model.score(x_train, y_train))
        return model
    else:
        return model, model.predict(x_test)


def logistic_regression(x_train, y_train, x_test = False):
    '''
    Can add penalty = 'l1' or 'l2', and change C to a smaller number(default 1)
    predict is used to produce labels. Can also use model.predict_proba to give the probability.
    get_params
    Options: polynomial regression, different data transformation, interaction, regularization.
    :param x_train:
    :param y_train:
    :param normalize:
    :param x_test:
    :return:
    '''
    model = linear_model.LogisticRegression()

    # Train the model using the training sets and check score
    model.fit(x_train, y_train)
    if x_test == False:
        print('Score:', model.score(x_train, y_train))
        return model
    else:
        return model, model.predict(x_test)


def decision_tree_classifier(x_train, y_train, criterion = 'gini', min_samples_split = 2, x_test = False):
    '''
    criterion can be 'gini' or 'entropy'
    can return model.feature_importances_, predict_proba
    :param x_train:
    :param y_train:
    :param criterion:
    :param x_test:
    :return:
    '''
    model = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split = min_samples_split)
    model.fit(x_train, y_train)
    if x_test == False:
        print('Score:', model.score(x_train, y_train))
        return model
    else:
        return model, model.predict(x_test)


def decision_tree_regressor(x_train, y_train, min_samples_split = 2, x_test = False):
    '''
    can return model.feature_importances_
    :param x_train:
    :param y_train:
    :param min_samples_split:
    :param x_test:
    :return:
    '''

    model = tree.DecisionTreeRegressor(min_samples_split = min_samples_split)
    model.fit(x_train, y_train)
    if x_test == False:
        print('Score:', model.score(x_train, y_train))
        return model
    else:
        return model, model.predict(x_test)


def svc(x_train, y_train, C = 1, kernel = 'rbf', x_test = False):
    '''
    C-Support Vector Classification. based on libsvm.  complexity is more than quadratic.
    hard to scale to dataset with more than a couple of 10000 samples.
    kernel can be:It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
    If a callable is given it is used to pre-compute the kernel matrix from data matrices;
     that matrix should be an array of shape (n_samples, n_samples).
    :param x_train:
    :param y_train:
    :param x_test:
    :return:
    '''
    model = svm.svc(C = C, kernel = kernel)
    model.fit(x_train, y_train)
    if x_test == False:
        print('Score:', model.score(x_train, y_train))
        return model
    else:
        return model, model.predict(x_test)


def naive_bayes(x_train, y_train, x_test = False):
    '''
    predict_proba(X)
    :param x_train:
    :param y_train:
    :param x_test:
    :return:
    '''
    model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes

    model.fit(x_train, y_train)
    if x_test == False:
        return model
    else:
        return model, model.predict(x_test)


def KNN(x_train, y_train, x_test = False, weights = 'uniform', k = 5):
    '''
    weights can be 'uniform', ‘distance’  or callable.
    can give predict_proba, kneighbors, kneighbors_graph
    note: compute cost high, should standardize the inputs and remove the outliers beforehand.
    :param x_train:
    :param y_train:
    :param x_test:
    :param weights:
    :param k:
    :return:
    '''
    model = KNeighborsClassifier(n_neighbors=k, weights = weights)
    # Train the model using the training sets and check score
    model.fit(x_train, y_train)
    if x_test == False:
        return model
    else:
        return model, model.predict(x_test)

def Kmeans(x_train, y_train, x_test = False, n_clusters = 3):
    '''
    get  cluster_centers_, labels_
    :param x_train:
    :param y_train:
    :param x_test:
    :param n_clusters:
    :return:
    '''
    model = KMeans(n_clusters=n_clusters, random_state=0)

    model.fit(x_train, y_train)
    if x_test == False:
        return model
    else:
        return model, model.predict(x_test)


def random_forest(x_train, y_train, x_test = False, n_estimators = 10):
    model = RandomForestClassifier(n_estimators = n_estimators)
    model.fit(x_train, y_train)
    if x_test == False:
        return model
    else:
        return model, model.predict(x_test)


def pca(x_train, x_test = False, n_components = 2):
    '''
    explained_variance_
    :param x_train:
    :param x_test:
    :param n_components:
    :return:
    '''
    model= decomposition.PCA(n_components=n_components)
    train_reduced = model.fit_transform(x_train)

    if x_test == False:
        return train_reduced
    else:
        test_reduced = model.transform(x_test)
        return train_reduced, test_reduced


def GD(x_train, y_train, x_test = False, n_estimators = 100):
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0, max_depth=1, random_state=0)
    model.fit(x_train, y_train)
    if x_test == False:
        return model
    else:
        return model, model.predict(x_test)