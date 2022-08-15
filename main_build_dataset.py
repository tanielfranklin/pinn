import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from BuildDataset import BuildingDataset
import pickle
import json
from pinn_BCS import pinn_vfm
from Logger import Logger
from utils import Struct, plot_result, prep_data_plot, store_model_files

# time = np.linspace(0, maxtime, 200) # Regular points inside the domain
# Treinamento e validação
dados = np.load('./dataset/BCS_data_train_limitado_f_zc_pm_pr_ident.npz')
dados_test = np.load('./dataset/BCS_data_train_limitado_f_zc_pm_pr_oper.npz')
# Somente Validação

# building dataset for training
n_steps_in, n_steps_out = 20, 1  # convert into input/output many-to-one
ds=BuildingDataset(n_steps_in, n_steps_out, dados, batch_size=500)
# y, train_dataset, test_y, test_X, train_X, train_y, u_train, _ = build_dataset(
#     n_steps_in, n_steps_out, dados, batch_size=500, parameters=par)
#plt.show()
file_name="dataset01.pk"

def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
save_object(ds,file_name)
# file_name="dataset01.json"
# with open(file_name, "w") as file_object:
#     json.dump(ds, file_object) 