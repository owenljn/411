from __future__ import print_function;
import numpy as np;
import scipy.io;
import itertools;
import random;
from sklearn import neighbors;

## Load data
data = scipy.io.loadmat('labeled_images.mat');
label = loadmat('val_labels.mat')
labels = label['val_labels'][:, 0];
xs = data['tr_images'].T;
del data;
# Preprocess images

xs = xs.reshape((-1, 1024)).astype('float');
xs -= np.mean(xs, axis=1)[:, np.newaxis];
xs /= np.sqrt(np.var(xs, axis=1) + 0.01)[:, np.newaxis];
# Make ids unique (for -1s)
inc = itertools.count(start=-1, step=-1).__iter__();
#uids = [id[0] if id != -1 else inc.next() for id in ids];
# Targets
emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];

########################################################################################################################
nfold = 10;

accuracy = {};
for k in range(1, 25):
    print("{}-nearest neighbors".format(k));
    acc = [];

    # # separating out people with the same identity
    # people = [(id, [i for i, x in enumerate(uids) if x == id]) for id in set(uids)];
    # # shuffling people
    # random.shuffle(people)

    # dividing people into groups of roughly the same size but not necessarily
    foldsize = (len(people) + nfold - 1) / nfold;
    foldids = [];
    for i in range(0, nfold):
        inds = [y for x in people[i * foldsize:(i + 1) * foldsize] for y in x[1]];
        foldids.append(inds);

    # perform nfold training and validation
    for i in range(1, nfold):
        print("Fold {}".format(i), end="");
        training_set = [];
        for e in range(len(foldids)):
            if e == i:
                continue;
            training_set.extend(foldids[e]);
        validation_set = foldids[i];

        clf = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm="brute");
        clf.fit(xs[training_set], labels[training_set]);
        acc.append(clf.score(xs[validation_set], labels[validation_set]))
        print(" - {}".format(acc[-1]));
    accuracy[k] = sum(acc) / float(len(acc));
    print("Average accuracy={}".format(accuracy[k]))
    print("");
print(sorted(accuracy.items(), key=lambda x:x[1]));