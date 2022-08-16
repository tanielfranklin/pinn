

import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from data.pinn_BCS import pinn_vfm

from data.Logger import Logger
from data.utils import Struct, load_model_data
from data.TrainingReport import TrainingReport


with open("dataset01.pk", 'rb') as open_file:
    ds = pickle.load(open_file)
# ========================================
# Creating the model and training
logger = Logger(frequency=100)
# logger.set_error_fn(error)
tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
neurons = 15
rho = 950
PI = 2.32*1e-9
start_rho = 0.9*950/rho
start_PI = 0.9*2.32*1e-9/PI
var = [start_rho, start_PI]  # normalized parameters
n_features = 6  # Network inputs  (fk, zc,pmc,pr, x1,x2)
# nt_config.maxIter = 100
Nc = 10
pinn = pinn_vfm(Nc, tf_optimizer, logger,
                var=var, pinn_mode="on",
                inputs=n_features,
                n_steps_in=ds.n_steps_in,
                n_steps_out=ds.n_steps_out,
                parameters=ds.parameters)

#######################################
pinn.lamb_l1 = tf.constant(1.0, dtype=tf.float32)  # x1 residue weight
pinn.lamb_l2 = tf.constant(1.0, dtype=tf.float32)  # x3 residue weight
pinn.lamb_l3 = tf.constant(1.0, dtype=tf.float32)  # x3 residue weight
# #######################################
local="model_adam_lbfgs/"
pinn.u_model.load_weights(local+'model.h5')


pinn_restored=restore_pinn_model()

training_report = TrainingReport(pinn, pinn_restored, ds)
plt.show() # Uncomment to see the graphics


# Uncomment the lines below to save the model
# folder_string="model_adam_200"
# objects2save={"Loss":loss_history,"trainstate":trainstate,"vartrain":var_history}
# save_model_files(folder_string,objects2save,pinn)