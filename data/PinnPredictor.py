import tensorflow as tf

class PinnPredictor(object):
    def __init__(self,pinn):
        #self.x0 = x0
        self.model = pinn.u_model
    def next_step(self,x):
        return self.model.predict(x)
    def many_steps(self,tstart,nsim,Xdata,ydata,udata):
        X=Xdata[tstart:tstart+1,:,:]
        y=ydata[tstart:tstart+nsim+1,:,:]
        u=udata[tstart:tstart+nsim+1,:,:]
        U=Xdata[tstart:tstart+nsim+1,:,:4]
        Xx=X[:,:,-2:]
        X0=X
        y0=self.model.predict(X0)[:,0:1,:]
        pred=y0
        for i in range(nsim):
            Xx=tf.concat([Xx[:,1:,:],y0[:,:,:-1]],1) # Remove o instante mais antigo e atualiza o vetor de estados com a nova predição (remove q) 
            #print(U[i:i+1,:,:].shape)
            X0=tf.concat([U[i:i+1,:,:],Xx],2) # Remonta o vetor de entrada da rede (exógenas+saidas)
            y0=model.predict(X0)[:,0:1,:]
            pred=tf.concat([pred,y0[:,0:1,:]],0)
        return pred,y,u
