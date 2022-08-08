from numpy import load as np_load
from numpy import save as np_save

from dataset_generator import generate_dataset
from model import fit_my_model

if __name__ == "__main__":
    #fake_path = "./fake/"
    fake_path= '/home/mathieu/Projetc2_Multi_Datasets/phase-01-training/dataset-dist/phase-01/training/fake/'
    X1, X2, Y = generate_dataset(ds_path=fake_path)

    np_save('X1.npy', X1)
    np_save('X2.npy', X2)
    np_save('Y.npy', Y)

    X1 = np_load('X1.npy')
    X2 = np_load('X2.npy')
    Y = np_load('Y.npy')

    X1_tr = X1[:1704]
    X1_val = X1[1704:1917]

    X2_tr = X2[:1704]
    X2_val = X2[1704:1917]

    Y_tr = Y[:1704]
    Y_val = Y[1704:1917]

    fit_my_model(X1_tr, X2_tr, Y_tr, X1_val, X2_val, Y_val)

    print('pass')
