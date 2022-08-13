from utils import add_noise_norm, plot_u, plot_states_BCS
import tensorflow as tf
import numpy as np
import pandas as pd

f_lim=(30,75)
qlim_m3=[15,65]
qlim=(qlim_m3[0]/3600,qlim_m3[1]/3600)

def Lim_c(xin):
    return xin[1]-xin[0]
F1c,F2c=8111.87,19468.5
F1lim,F2lim=(99811.5, 107923),(239548,259016)
Hc=1557.0851455268821
qcc=0.03290348005910621

zclim=(0,1)
pmlim=(1e5,50e5)
pbhlim=(1e5,1.26e7) 
pwhlim=(1e5,50e5) 

pbc=Lim_c(pbhlim)
pwc=Lim_c(pwhlim)
qc=Lim_c(qlim)
pbmin=pbhlim[0]
pwmin=pwhlim[0]
qmin=qlim[0]
H_lim=(-136.31543417849096, 1420.7697113483912)
qch_lim=(0.0, 0.03290348005910621)
#qch_lim=(0.0, 0.03290348005910621)

#rho=tf.Variable(9.0)*100 #PI = 2.32e-9; # Well productivy index [m3/s/Pa]
#PI = tf.Variable(2.15)*1e-9
xc=np.array([pbc,pwc,qc])
x0=np.array([pbmin,pwmin,qmin])
# x=np.hstack(dataset_full[0:3])
# xtest=np.array(dataset_test[0:3])
def normalizar_x(x,xc,x0):
    xn=[(x[:,i]-x0[i])/xc[i] for i in range(3)]
    return np.array(xn).T

# Normalizing factors
prc,pr0=(1.4e7-1.2e7),1.2e7
pm_c,pm0=(2.1e6-1.2e6),1.2e6
def normalizar_u(u,fator):
    aux=[]
    u[2]=u[2]-pm0 
    u[3]=u[3]-pr0
    for i,valor in enumerate(u):
        aux.append(valor/fator[i])
    return np.hstack(aux)

# xn=normalizar_x(x,xc,x0)
# xtestn=normalizar_x(np.hstack(dataset_test[0:3]),xc,x0)
def split_data(n_in,n_out,data):
    a,b,c=split_sequences(data, n_in, n_out)
    x=a[:,:,:]
    y=b[:,:,-3:]
    u_train=b[:,:,0:4] #catch 0 1 and 2 values
    return x,y,u_train

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    #https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
 X, y, u = list(), list(),list()
 for i in range(len(sequences)):
  # find the end of this pattern
  end_ix = i + n_steps_in
  out_end_ix = end_ix + n_steps_out-1
  # check if we are beyond the dataset
  if out_end_ix > len(sequences)-1:
   break
  # gather input and output parts of the pattern
  seq_x, seq_y, seq_u= sequences[i:end_ix, :], sequences[end_ix:out_end_ix+1, :],sequences[end_ix-1:out_end_ix, :]
  X.append(seq_x)
  y.append(seq_y)
  u.append(seq_u)
 return np.array(X), np.array(y), np.array(u)# choose a number of time steps #change this accordingly



def build_dataset(n_steps_in, n_steps_out,dados,batch_size):
    def reshape_data(dataset,length):
        dataset_new=[]
        for i in dataset:
            dataset_new.append(i.reshape([length,1]))
        return dataset_new
    
    
    fk=dados['U'][:,0:1]
    zc=dados['U'][:,1:2]
    x1=dados['x1']
    x2=dados['x2']
    x3=dados['x3']
    pmc=dados['U'][:,2:3]
    pr=dados['U'][:,3:4]
    tempo=dados['t']
    maxtime = fk.shape[0]
    nsim=maxtime
    dataset_full=[x1,x2,x3,fk,zc,pmc,pr,tempo]
    dataset_full=reshape_data(dataset_full,nsim)
    x1,x2,x3,fk,zc,pmc,pr,tempo=dataset_full

        #------------------------------------------------
    ### Inserindo ruido
    sigma=[0.01,0.01,0.01,0.005,0.001,0.01,0.01]
    dataset_full_noisy=[]
    for i,d in enumerate(dataset_full[0:-1]):
        dataset_full_noisy.append(add_noise_norm(d,sigma[i]))
    dataset_full_noisy.append(tempo)
    dataset_full=dataset_full_noisy
    #----------------------------------------------------



    # dataset_full=[x1,x2,x3,fk,zc,pmc,pr,tempo]
    # dataset_full=reshape_data(dataset_full,nsim)
    x1,x2,x3,fk,zc,pmc,pr,tempo=dataset_full
    #------------------------------------------------
    # # Reducing dataset size
    nsim=maxtime
    dataset_limited=[]
    for i in dataset_full:    
        dataset_limited.append(i[0:nsim,:])
    dataset_full=dataset_limited
    x1,x2,x3,fk,zc,pmc,pr,tempo=dataset_limited
    ts=1
    x=np.hstack(dataset_full[0:3])
    uplot=dataset_full[3:-1]
    Fig_u=plot_u(uplot)
    Fig_x=plot_states_BCS(x,tempo)
    xn=normalizar_x(x,xc,x0)
    print("Limites das exógenas")
    for i in dataset_full[3:7]:
        print(f"Max:{max(i)}, Min: {min(i)}")
    u=np.hstack(dataset_full[3:7])
    un=normalizar_u(dataset_full[3:7],[60,100,pm_c,prc])
    Figs={}
    Fig_xn=plot_states_BCS(xn,tempo,norm=True)
    uplot=[un[:,i] for i in range(4)]
    Fig_un=plot_u(uplot)
    Figs["un"]=Fig_un
    Figs["xn"]=Fig_xn
    df = pd.DataFrame(np.hstack([un,xn]),columns=['fn','zn','pmn','prn','pbh','pwh','q'])
    df_u = pd.DataFrame(np.hstack([u]),columns=['f','z','pm','pr'])
    dset = df.values.astype(float)
    du_set = df_u.values.astype(float)
    #dset_test=df_test.values.astype(float)
    X,y,u_train=split_data(n_steps_in, n_steps_out,dset)
    split_point = int(0.7*dset.shape[0])
    split_point = int(0.98*dset.shape[0])
    train_X_full , train_y_full, u_train = X[:split_point, :] , y[:split_point, :], u_train[:split_point, :]
    test_X_full , test_y_full = X[split_point:, :] , y[split_point:, :]
    uk=dset[0:split_point,0:4]
    #Remove unmeasured variable q from training dataset
    train_y=train_y_full[:,:,0:2]
    train_X=train_X_full[:,:,:-1]
    test_y=test_y_full[:,:,0:2]
    test_X=test_X_full[:,:,:-1]
    uk=tf.convert_to_tensor(uk, dtype=tf.float32) # u(k) para ODE
    train_X=tf.convert_to_tensor(train_X, dtype=tf.float32) # X(k) para ODE
    train_y=tf.convert_to_tensor(train_y, dtype=tf.float32) # y(k) para ODE
    train_y_full=tf.convert_to_tensor(train_y_full, dtype=tf.float32) # y(k) para ODE
    u_train=tf.convert_to_tensor(u_train, dtype=tf.float32) # y(k) para ODE
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X,train_y, u_train))
    train_dataset = train_dataset.batch(batch_size)
   
    Figs["u"]=Fig_u
    Figs["x"]=Fig_x
    return y,train_dataset,test_y,test_X,train_X, train_y_full, u_train, Figs

