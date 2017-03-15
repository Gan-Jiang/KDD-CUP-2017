import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve, GridSearchCV
import sklearn.preprocessing as preprocessing
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingRegressor, BaggingClassifier


#test data
from sklearn.datasets import make_classification
X, y = make_classification(1000, n_features=20, n_informative=2,
                           n_redundant=2, n_classes=2, random_state=0)
from pandas import DataFrame
df = DataFrame(np.hstack((X, y[:, None])))


def plot_scatter(df, label_index = None):
    '''
    plot scatter plot of pd.dataframe
    :param df:
    :param label_index: the index of the label. Used only for classification problem.
    :return:
    '''
    _ = sns.pairplot(df[:50], vars=[8, 11, 12, 14, 19], hue=label_index, size=1.5)
    plt.show()


def plot_corr(df):
    '''
    plot the diagonal corrlation matrix
    :param df: pd.dataframe
    :return:
    '''
    sns.set(style="white")

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.show()


def plot_learning_curve(estimator, X, y, cv=5, ylim=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    '''
    Plot the learning curve of the data on one model.
    :param estimator: object type that implements the “fit” and “predict” methods
    :param X: input features. Numpy format
    :param y: label
    :param cv: the number of folds in the cross validation.
    :param ylim: tuple(ymin, ymax) in order to plot
    :param train_sizes: x axis value
    :return:
    '''

    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title('Learning Curve')
    plt.show()


def grid_search(estimator, param_grid, X, y):
    '''
    :param estimator:
    :param param_grid: param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    :param X: input features
    :param y: labels
    :return: gridsearch object. Can call estm.best_params_ or best_estimator_
    '''
    estm = GridSearchCV(estimator, param_grid=param_grid)
    estm.fit(X, y)


def plot_feature_hist(feature, ylabel, title):
    '''
    :param feature: Series
    :param ylabel:
    :param title:
    :return:
    '''
    feature.value_counts().plot(kind="bar")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


'''
Stack hist
fig = plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'Unsurvived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"aaa")
plt.xlabel(u"Pclass")
plt.ylabel(u"Number")
plt.show()
'''


'''
when #class too large
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)
'''


def covert_dummy(df, feature_name, prefix):
    '''
    :param df:
    :param feature_name: the name of the feature we want to do one hot encoding.
    :param prefix:
    :return:
    '''
    dummies_feature = pd.get_dummies(df[feature_name], prefix=prefix)
    df = pd.concat([df, dummies_feature], axis=1)
    df.drop([feature_name], axis=1, inplace=True)
    return df


def standardize(df, feature_name):
    scaler = preprocessing.StandardScaler()
    scale_param = scaler.fit(df[feature_name])
    try:
        df[feature_name + '_scaled'] = scaler.fit_transform(df[feature_name], scale_param)
    except:
        df[str(feature_name) + '_scaled'] = scaler.fit_transform(df[feature_name], scale_param)


'''
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
'''


def bagging_reg(estimator, X, y, n_estimators = 20, max_samples = 0.8, max_features = 1, bootstrap = True, bootstrap_features = False):
    bagging_estimator = BaggingRegressor(estimator, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, bootstrap=bootstrap,
                                   bootstrap_features=bootstrap_features, n_jobs=-1)
    bagging_estimator.fit(X, y)
    return bagging_estimator


def bagging_clf(estimator, X, y, n_estimators = 20, max_samples = 0.8, max_features = 1, bootstrap = True, bootstrap_features = False):
    bagging_estimator = BaggingClassifier(estimator, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, bootstrap=bootstrap,
                                   bootstrap_features=bootstrap_features, n_jobs=-1)
    bagging_estimator.fit(X, y)
    return bagging_estimator