
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
from data.pinn_BCS import pinn_vfm
from data.Logger import Logger
from data.utils import restore_pinn_model, plot_test
from data.TrainingReport import TrainingReport
from data.PinnPredictor import PinnPredictor


with open("dataset_opera.pk", 'rb') as open_file:
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
local = "model_adam_lbfgs/"
#local = "model_adam_lbfgs/"
pinn.u_model.load_weights(local+'model.h5')
pinn_restored = restore_pinn_model(local)
######################################
# training_report = TrainingReport(pinn, pinn_restored, ds)
# training_report.gen_plot_result()
# training_report.gen_var_plot()
# training_report.gen_plot_loss_res()
predictor=PinnPredictor(pinn)
ds.train_X
#print(ds.train_X[0:1,:,:])
x0=ds.train_X[0:1]


# for i in range(3):
#     y=predictor.next_step(x0)
#     x0=y
#     print(y)

def free_predict(tstart,nsim,ds,model):
    X=ds.train_X[tstart:tstart+1,:,:]
    y=ds.train_y_full[tstart:tstart+nsim+1,:,:]
    u=ds.train_X[tstart:tstart+nsim+1,:,:]
    U=ds.train_X[tstart:tstart+nsim+1,:,:4]
    Xx=X[:,:,-2:]
    X0=X
    y0=model.predict(X0)[:,0:1,:]
    pred=y0
    for i in range(nsim):
        Xx=tf.concat([Xx[:,1:,:],y0[:,:,:-1]],1) # Remove o instante mais antigo e atualiza o vetor de estados com a nova predição (remove q) 
        #print(U[i:i+1,:,:].shape)
        X0=tf.concat([U[i:i+1,:,:],Xx],2) # Remonta o vetor de entrada da rede (exógenas+saidas)
        y0=model.predict(X0)[:,0:1,:]
        pred=tf.concat([pred,y0[:,0:1,:]],0)
    return pred,y,u
    
tstart=310
nsim=300
pred, yreal, ureal=free_predict(tstart,nsim,ds,predictor.model)
#np.square(yreal[:,0,2]- pred[:,0,2])
FigFree=plot_test(yreal, pred,[ds.parameters.xc,ds.parameters.x0])

plt.show()  # Uncomment to see the graphics
