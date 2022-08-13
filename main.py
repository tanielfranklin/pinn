import numpy as np
from BuildDataset import build_dataset
#time = np.linspace(0, maxtime, 200) # Regular points inside the domain
#Treinamento e validação
dados = np.load('./dataset/BCS_data_train_limitado_f_zc_pm_pr_ident.npz')
dados_test = np.load('./dataset/BCS_data_train_limitado_f_zc_pm_pr_oper.npz')
#Somente Validação
#building dataset for training 
n_steps_in, n_steps_out = 20 ,1# convert into input/output many-to-one
X_train, y_train, u_train=build_dataset(n_steps_in, n_steps_out,dados)