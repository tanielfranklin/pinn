import tensorflow as tf
import numpy as np

def test_res(y,u,model):
    model.rho=tf.Variable(1.0, dtype=tf.float32)
    model.PI=tf.Variable(1.0, dtype=tf.float32)
    r1,r2,r3=model.ED_BCS(y,u)
    return r1.numpy(),r2.numpy(),r3.numpy()

def dydt2(y_pred,ts):
    #Central 3 pontos
    y = y_pred[:,0,:]
    n=y.shape[0]
    try:
        if n<6:
            raise Exception("Model output size must have at least 6 time points ")          
    except Exception as inst:
        print(inst.args)
        raise
    #Progressiva e regressiva 3 pontos
    pro3=tf.constant([[-3,4,-1]],dtype=tf.float32)/(2*ts)
    reg3=tf.constant([[1,-4,3]],dtype=tf.float32)/(2*ts)
    d1=tf.matmul(pro3,y[0:3,:])
    #print(d1)
    dn=tf.matmul(reg3,y[-3:,:])
    #Central 2 pontos
    dc=(y[2:n,:]-y[0:n-2,:])/(2*ts)        
    return tf.concat([d1,dc,dn],axis=0)


def plot_states_BCS(input,t,norm=False): 
    scale=np.array([1/1e5,1/1e5,3600])
    if norm==True:
        scale=np.array([1,1,1])
    fig4=plt.figure()
    label = ['Pbh','Pbw','q'];
    for i,val in enumerate(label): 
        ax1=fig4.add_subplot(len(label),1,i+1)   
        ax1.plot(tempo ,input[:,i]*scale[i], label=val)
        if i!=2:
            ax1.set_xticklabels([])
        ax1.set_ylabel(val)
        plt.grid(True)
    return fig4
def plot_u(uplot):
    fig=plt.figure()
    label = ['f','z','pman','pr'];
    for i,val in enumerate(label):
        ax=fig.add_subplot(len(label),1,i+1)
        ax.plot(uplot[i])#, label=val)
        ax.set_ylabel(val)
        plt.grid(True)
    return fig

def add_noise_norm(signal,sigma):
    n = np.random.normal(0, sigma, len(signal))
    n=n.reshape([len(n),1])
    sig=(1+n)*signal
    return sig

def add_noise(signal,snr_db):
  # Set a target SNR
  target_snr_db = snr_db
  # Calculate signal power and convert to dB 
  sig_avg_watts = np.mean(signal**2)
  sig_avg_db = 10 * np.log10(sig_avg_watts)
  # Calculate noise according to [2] then convert to watts
  noise_avg_db = sig_avg_db - target_snr_db
  noise_avg_watts = 10 ** (noise_avg_db / 10)
  # Generate an sample of white noise
  mean_noise = 0
  noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
  # Noise up the original signal
  noise_volts=noise_volts.reshape([len(signal),1])
  return signal + noise_volts




# loc_drive='/content/drive/MyDrive/Dados_BCS/'

# import sys
# sys.path.append('/drive/MyDrive/Dados_BCS/')


# exec(compile(open(loc_drive+'subrotinas.py', "rb").read(), loc_drive+'subrotinas.py', 'exec'))
Font=14 # size of plot text
def store_model_files(file_name_list,obj_list): 
    for i in np.arange(len(file_name_list)):
        with open(file_name_list[i], "wb") as open_file:
            pickle.dump(obj_list[i], open_file)
   
    print("Saved files")

def load_model_data(file_name):
    with open(file_name, 'rb') as open_file:
        obj = pickle.load(open_file)
    return obj

def plot_test(yp, obs, u_test,step):
        Font=14
        erro=tf.square(obs[:,:,:] - yp[:,:,0:3])
        yp=yp[:,step,:]*xc+x0
        obs=obs[:,step,:]*xc+x0
        u_test=u_test[:,step,:]
        k=np.arange(0,yp.shape[0])/60
        Fig=plt.figure(figsize=(10, 9))
        sc=[1/1e5, 1/1e5,3600]
        sc_u=[60, 100,1/1e5,1/1e5]
        uc=[60,100,pm_c,prc]
        u0=[0,0,pm0,pr0]
        scu=[1,1,1/1e5,1/1e5]
        
        label=["$P_{bh}(bar)$","$P_{wh}(bar)$", "$q (m^3/h)$"]
        label_u = ['$f(Hz)$',r'$Z_c$(\%)', "$p_{m} (bar)$","$p_{r} (bar)$"];
        for i,lb in enumerate(label):        
            ax1=Fig.add_subplot(len(label+label_u),1,i+1)
            ax1.plot(k, obs[:,i]*sc[i],"-k", label='GroundTruth')
            ax1.plot(k, yp[:,i]*sc[i],":",color='blue',lw=2,label=f"Prediction (MSE: {tf.reduce_mean(erro).numpy():0.3f})")
            ax1.set_ylabel(lb,  fontsize=Font)
            ax1.set_xticklabels([])
            
            ax1.grid(True)
            plt.setp(ax1.get_yticklabels(), fontsize=Font)  
        plt.legend(bbox_to_anchor=(1, 3.5), ncol = 2,fontsize=Font)
        #plt.legend()
        for i,lb in enumerate(label_u):
            ax1=Fig.add_subplot(len(label+label_u),1,i+1+3)
            ax1.plot(k, (u_test[:,i]*uc[i]+u0[i])*scu[i],"-k")
            ax1.set_ylabel(lb,  fontsize=Font)
            ax1.grid(True)
            if i!=3:
                ax1.set_xticklabels([])
            plt.setp(ax1.get_yticklabels(), fontsize=Font)  
        ax1.set_xlabel('$Time (min)$' ,  fontsize=Font)
        plt.setp(ax1.get_xticklabels(), fontsize=Font)
             
        return Fig

def plot_result(pred_train, pred_test, obs):   
    if obs.shape[-1]==2:
        obs=obs*xc[0:2]+x0[0:2]
    if obs.shape[-1]==3:
        obs=obs*xc+x0

    #y_mean = np.mean(prediction1)
    
    k = np.arange(0,len(obs))
    ktr=np.arange(0,len(pred_train))
    kts=np.arange(len(pred_train),len(pred_train)+len(pred_test))
    #print(k.shape,ktr.shape,kts.shape)
    Fig=plt.figure(figsize=(10, 4))
    label=["$P_{bh}(bar)$","$P_{wh}(bar)$","$q (m^3/h)$"]
    scale=[1/1e5,1/1e5,3600]
    cor=["black","black","gray"]
    leg_lb=['Observed data','Observed data','No measured data']


    for i,val in enumerate(label):
        ax1=Fig.add_subplot(len(label),1,i+1)
        if i==2:
            l0, =ax1.plot(k, obs[:,i]*scale[i],"-",color=cor[i])
        else:
            l1, =ax1.plot(k, obs[:,i]*scale[i],"-",color=cor[i])

        l2, =ax1.plot(ktr, pred_train[:,i]*scale[i],":",color='red',lw=2)
        l3, =ax1.plot(kts, pred_test[:,i]*scale[i],":",color='blue',lw=2)
        ax1.set_ylabel(val,  fontsize=Font)
        plt.setp(ax1.get_yticklabels(), fontsize=Font)
        if i!=2:
            ax1.set_xticklabels([])
        
        ax1.grid(True)
    plt.grid(True)
    ax1.set_xlabel('$Time(s)$' ,  fontsize=Font)
    plt.setp(ax1.get_xticklabels(), fontsize=Font)
    plt.setp(ax1.get_yticklabels(), fontsize=Font)
    plt.legend([l0,l1,l2,l3],['Non measured data','Observed data','Prediction with training data','Prediction with validation data'],bbox_to_anchor=(1.0, 4.2), ncol = 2,fontsize=Font)
    #plt.legend(bbox_to_anchor=(1, 3.8), ncol = 2)
    #fig.legend(handles=[l1, l2])
    return Fig



# Plot history and future
def plot_multistep(history, prediction1 , groundtruth , start , end):
    plt.figure(figsize=(20, 4))
    y_mean = np.mean(prediction1)
    range_history = len(history)
    range_future = list(range(range_history, range_history + len(prediction1[:,0])))
    Fig=plt.figure()
    #plt.title("Test Data from {} to {} , Mean = {:.2f}".format(start, end, y_mean) ,  fontsize=18)
    ax1=Fig.add_subplot(3,1,1)
    ax1.plot(np.arange(range_history), np.array(history[:,0]/1e5), label='History')
    ax1.plot(range_future, np.array(prediction1[:,0]/1e5),label='Forecasted with LSTM')
    ax1.plot(range_future, np.array(groundtruth[:,0]/1e5),":k",label='GroundTruth')
    ax1.set_ylabel("Pbh",  fontsize=Font)
    ax1.set_xticklabels([])
    ax1.grid(True)
    plt.grid(True)
    ax2=Fig.add_subplot(3,1,2)
    ax2.set_ylabel("Pwh",  fontsize=Font)
    ax2.plot(np.arange(range_history), np.array(history[:,1]/1e5), label='History')
    ax2.plot(range_future, np.array(prediction1[:,1]/1e5),label='Forecasted')
    ax2.plot(range_future, np.array(groundtruth[:,1]/1e5),":k",label='Observed')
    ax2.set_xticklabels([])
    ax2.grid(True)
    plt.grid(True)
    ax3=Fig.add_subplot(3,1,3)
    ax3.set_ylabel("q", fontsize=Font)
    ax3.plot(np.arange(range_history), np.array(history[:,2]*3600), label='History')
    ax3.plot(range_future, np.array(prediction1[:,2]*3600),label='Forecasted')
    ax3.plot(range_future, np.array(groundtruth[:,2]*3600),":k",label='Observed')
    plt.grid(True)
    #plt.legend(loc='upper left')    
    ax2.set_xlabel('Time(s)' ,  fontsize=Font)
    plt.legend(bbox_to_anchor=(0.9, -0.2), ncol = 3)

# Calculate MAE and RMSE
def evaluate_prediction(predictions, actual, model_name , start , end):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()    
    print("Test Data from {} to {}".format(start, end))
    print('Mean Absolute Error: {:.3f}'.format(mae))
    print('Root Mean Square Error: {:.3f}'.format(rmse))
    print('')
    print('')


def prep_data_plot(model,train_X, train_y , test_X , test_y):  
    y_pred_train=model.predict(train_X)
    y_pred_train=y_pred_train.reshape(y_pred_train.shape[0]*y_pred_train.shape[1],y_pred_train.shape[2])
    y_pred_test=model.predict(test_X)
    y_pred_test=y_pred_test.reshape(y_pred_test.shape[0]*y_pred_test.shape[1],y_pred_test.shape[2])
    for i in range(y_pred_train.shape[0]):
        k=i*n_steps_out
        if k==0:
            pred_train=y_pred_train[k:k+1,:]
        pred_train=np.vstack((pred_train,y_pred_train[k:k+1,:]))
        if k==y_pred_train.shape[0]:
            #print(k)
            break
    
    for i in range(y_pred_test.shape[0]):
        k=i*n_steps_out
        if k==0:
            pred_test=y_pred_test[k:k+1,:]
        pred_test=np.vstack((pred_test,y_pred_test[k:k+1,:]))
        if k==y_pred_train.shape[0]:
            break
    return pred_train*xc+x0, pred_test*xc+x0

def prep_data(model,x_test, y_test , start , end , last):
    #prepare test data X
    dataset_test = x_test
    dataset_test_X = dataset_test[start:end, :]
    print("dataset_test_X :",dataset_test_X.shape)
    test_X_new = dataset_test_X.reshape(1,dataset_test_X.shape[0],dataset_test_X.shape[1])
    print("test_X_new :",test_X_new.shape)#prepare past and groundtruth
    npast=end-5*n_steps_in
    past_data = y_test[npast:end , :]
    dataset_test_y = y_test[end:last , :]
    print("dataset_test_y :",dataset_test_y.shape)
    print("past_data :",past_data.shape)#predictions
    y_pred = model.predict(test_X_new)
    y_pred_inv = y_pred
    y_pred_inv = y_pred_inv.reshape(n_steps_out,3)
    y_pred_inv = y_pred_inv[:,:]
    print("y_pred :",y_pred.shape)
    print("y_pred_inv :",y_pred_inv.shape)
    
    return y_pred_inv , dataset_test_y , past_data#start can be any point in the test data (1258)

@tf.function
def dydt(y_pred,ts):
    #Central 4 pontos
    y = y_pred
    n=y.shape[1]
    try:
        if n<6:
            raise Exception("Model output size must have at least 6 time points ")          
    except Exception as inst:
        print(inst.args)
        raise
    #Progressiva e regressiva 3 pontos
    pro3=tf.constant([[-3,4,-1]],dtype=tf.float32)/(2*ts)
    reg3=tf.constant([[1,-4,3]],dtype=tf.float32)/(2*ts)
    d1=tf.matmul(pro3,y_pred[:,0:3,:])
    dn=tf.matmul(reg3,y_pred[:,-3:,:])
    #Central 2 pontos
    dc=(y_pred[:,2:n,:]-y_pred[:,0:n-2,:])/(2*ts)        
    return tf.concat([d1,dc,dn],axis=1)





#@tf.function
def get_abs_max_grad(grad):
    r=np.zeros((len(grad))).astype(np.float32)
    for i in range(len(grad)):
        r[i]=tf.math.reduce_max(tf.abs(grad[i]))
    return tf.math.reduce_max(r)
#@tf.function
def get_abs_mean_grad(grad):
    r=np.zeros((len(grad))).astype(np.float32)
    for i in range(len(grad)):
        r[i]=tf.math.reduce_mean(tf.abs(grad[i]))
    return tf.math.reduce_mean(r)


# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)


def gen_traindata(file):
	data = np.load(file)
	return data["t"], data["x"], data["u"]
# time points

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value[0]] # concatena uma lista de	X com yhat (value)
	array = np.array(new_row) # converte para array
	array = array.reshape(1, len(array)) # Faz a transposta
	inverted = scaler.inverse_transform(array)	# reescala
	return inverted[0, -1] # retorna yhat (value) reescalonado

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

def forecast_on_batch(model, batch_size, X):
	X = X.reshape(batch_size, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]
