import numpy as np
import qml
import glob
import random
from qml.math import cho_solve
from qml.kernels import laplacian_kernel

def get_properties(filename, atype = "all"):
    with open(filename) as f:
        lines = f.readlines()
        y = []
        for line in lines[2:]:
            if "^" in line:
                print(filename)
                return 1
            tokens = line.split()
            if atype == "all" or tokens[0] == atype:
                y.append(float(tokens[-1]))
        return y

def get_descriptor_and_property(filenames, atype, cutoff):
    X = []
    Y = []
    for filename in filenames:
        # Get the partial charges (property to predict)
        y = get_properties(filename, atype)
        # generate a Compound data structure
        mol = qml.Compound(filename)
        # generate the descriptor
        mol.generate_atomic_coulomb_matrix(central_cutoff = cutoff, size = 30)
        # either use all atoms, or just atoms of a specific type
        if atype == "all":
            x = mol.representation
        else:
            x = mol.representation[mol.nuclear_charges == qml.data.NUCLEAR_CHARGE[atype]]


        # add the atoms to be used in this molecule to the entire set
        X.extend(x)
        Y.extend(y)
    return np.asarray(X), np.asarray(Y)

def calc_mae(y1, y2):
    return sum((y1-y2)**2) / len(y1)

# Limit to a specific atom type or "all" for everything
atype = "C"

filenames = glob.glob("qm9/*.xyz")
# shuffle the filenames since the naming is by molecule size
random.shuffle(filenames)

# Use 500 molecules for training
train_filenames = filenames[:500]
# Use 250 molecules for test
test_filenames = filenames[500:750]


# hyper parameters
sigmas = [1.0, 10.0, 10.0**2, 10.0**3]
cutoffs = [2.0, 3.0, 4.0]
llambda = 1e-8 # doesn't usually need to be changed

# try 3 different cutoffs
for cutoff in cutoffs:
    train_x, train_y = get_descriptor_and_property(train_filenames, atype, cutoff)
    test_x, test_y = get_descriptor_and_property(test_filenames, atype, cutoff)
    # in this case try out 4 different values of sigma
    for sigma in sigmas:
        # Get the kernel between all descriptors in the training set
        K = laplacian_kernel(train_x, train_x, sigma) + llambda * np.identity(train_x.shape[0])

        # get the KRR prefactors, i.e. this is the training of the network
        alpha = cho_solve(K, train_y)

        # get the kernel between all descriptors in the training set and all in the test set
        Ks = laplacian_kernel(test_x, train_x, sigma)

        # predict values of y
        y_pred = np.dot(Ks, alpha)

        print("predicted MAE of %.4f for sigma: %.4g, cutoff: %.1f and %d training points" % (calc_mae(y_pred, test_y), sigma, cutoff, len(train_x)))



