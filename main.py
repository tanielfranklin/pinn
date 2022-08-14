import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from BuildDataset import build_dataset
from pinn_BCS import pinn_vfm
from parameters import parameters
from Logger import Logger
from utils import Struct, plot_result, prep_data_plot
#time = np.linspace(0, maxtime, 200) # Regular points inside the domain
#Treinamento e validação
dados = np.load('./dataset/BCS_data_train_limitado_f_zc_pm_pr_ident.npz')
dados_test = np.load('./dataset/BCS_data_train_limitado_f_zc_pm_pr_oper.npz')
#Somente Validação
par=parameters()

#building dataset for training 
n_steps_in, n_steps_out = 20 ,1# convert into input/output many-to-one
y,train_dataset,test_y,test_X,train_X, train_y, u_train,_=build_dataset(n_steps_in, n_steps_out,dados,batch_size=500,parameters=par)
#plt.show()

#========================================
# # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
nt_config = Struct()
#Positive integer. The number of iterations allowed to run in parallel. 
nt_config.parallel_iter=2
#The maximum number of iterations for L-BFGS updates. 
nt_config.maxIter = 400
#Specifies the maximum number of (position_delta, gradient_delta) correction pairs to keep as implicit approximation of the Hessian matrix. 
nt_config.nCorrection = 50
#If the relative change in the objective value between one iteration and the next is smaller than this value, the algorithm is stopped. 
nt_config.tolfun=1e-5
#Specifies the gradient tolerance for the procedure. If the supremum norm of the gradient vector is below this number, the algorithm is stopped. 
nt_config.tol = 1e-5 #Specifies the gradient tolerance for the procedure. If the supremum norm of the gradient vector is below this number, the algorithm is stopped. 

##---------------------------------------

#========================================
# Creating the model and training
logger = Logger(frequency=100)
#logger.set_error_fn(error)
tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
neurons=15
rho=950
PI=2.32*1e-9
start_rho=0.9*950/rho
start_PI=0.9*2.32*1e-9/PI
var=[start_rho, start_PI] # parâmetros normalizados
#var=[1.0, start_PI] # Problema direto
#var=[start_rho, 1.0] # Problema direto
#var=[1.0, 1.0] # Problema direto
n_features=6 # Network inputs  (fk, zc,pmc,pr, x1,x2)
nt_config.maxIter = 100
Nc=10
pinn = pinn_vfm(Nc,tf_optimizer, logger,
                var=var,pinn_mode="on", 
                inputs=n_features, 
                n_steps_in=n_steps_in,
                n_steps_out=n_steps_out,
                parameters=par)

#######################################
pinn.lamb_l1=tf.constant(1.0, dtype=tf.float32) #x1 residue weight
pinn.lamb_l2=tf.constant(1.0, dtype=tf.float32) #x3 residue weight
pinn.lamb_l3=tf.constant(1.0, dtype=tf.float32) #x3 residue weight
# #######################################

Loss, trainstate,vartrain=pinn.fit(train_dataset, tf_epochs=10)#,adapt_w=True)                                  
pred_train,pred_test=prep_data_plot(pinn.u_model,train_X, train_y , test_X , test_y,par.xc,par.x0)
pinn.u_model.reset_metrics()
plot_result(pred_train, pred_test, y[:,0,:],par.xc,par.x0)
vartrain.plot_var()
Loss.plot_loss_res()