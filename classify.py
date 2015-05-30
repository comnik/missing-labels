import sys
import csv
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, metrics, cross_validation, semi_supervised


def to_feature_vec(row):
    """
    Returns the feature-vector representation of a piece of input data.
    """

    return [float(x) for x in row]

def get_features(inpath):
    """
    Reads our input data from a csv file.
    """

    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        X = [to_feature_vec(row) for row in reader]

    return np.atleast_1d(X)

def output_predictions(classifier, infile, outfile):
    """
    Loads features from an input file, and outputs predictions to outfile.
    """

    X = get_features(infile)
    Ypred = classifier.predict(X)
    np.savetxt(outfile, Ypred, delimiter=",", fmt="%i") # the last parameter converts the floats to ints

def snapshot(model, snappath):
    """
    Persists a model to disk.
    """

    with open(snappath, 'wb') as fout:
        pickle.dump(model, fout)

def load_snapshot(snappath):
    """
    Returns a model that was persisted to disk.
    """

    with open(snappath, 'rb') as fin:
        return pickle.load(fin)

def scorer(Ypred, Ytruth):
    """
    Calculates a measure of fit for given predictions.
    """

    return -np.log(max(0.0001, np.max(Ypred[np.where(Ytruth != -1)])))

def main(argv):
    plt.ion()

    # Read labelled training data.
    print('Loading data...')

    if '--reload' in argv:
        X = get_features('project_data/train.csv')
        Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')

        snapshot(X, 'snapshot/X.obj')
        snapshot(Y, 'snapshot/Y.obj')

    else:
        X = load_snapshot('snapshot/X.obj')
        Y = load_snapshot('snapshot/Y.obj')

        X = X[:20000]
        Y = Y[:20000]

    print('Finished.')

    # Cross validation.
    print('Training classifier...')

    classifier = semi_supervised.LabelPropagation()

    if '--reclassify' in argv:
        if '--cv' in argv:
            scorefun = metrics.make_scorer(scorer)
            scores = cross_validation.cross_val_score(classifier, X, Y, scoring=scorefun, cv=5)

            print('Mean: %s +/- %s' % (np.mean(scores), np.std(scores)))

        else:
            classifier.fit(X, Y)

        snapshot(classifier, 'snapshot/classifier.obj')

    else:
        classifier = load_snapshot('snapshot/classifier.obj')

    print('Finished.')

    # Output.
    if '--validate' in argv:
        output_predictions(classifier, 'project_data/validate.csv', 'out/validate_y.csv')

    if '--test' in argv:
        output_predictions(classifier, 'project_data/test.csv', 'out/test_y.csv')

    input('Press any key to exit...')


if __name__ == "__main__":
    main(sys.argv[1:])
