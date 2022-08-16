import numpy as np
import matplotlib.pyplot as plt

Font=14
class TrainingReport():
    def __init__(self,model,training_data,ds):
        self.loss,trainstate,var_history=training_data
        self.pack_plot=ds.pack_plot
        self.x0=ds.parameters.x0
        self.xc=ds.parameters.xc
        self.loss_fig=self.plot_loss_res()
        self.var_fig=var_history.plot_var()
        self.obs=ds.pack[0][:,0,:]
        self.pred_train, self.pred_test=self.__prep_data_plot(model.u_model) 
        self.result_fig=self.gen_plot_result()
    
    
    def plot_loss_res(self):

        loss_train = self.loss.loss_train
        loss_train_bc = self.loss.loss_train_bc
        loss_train_f = self.loss.loss_train_f
        loss_train_x1 = self.loss.loss_train_x1
        loss_train_x2 = self.loss.loss_train_x2
        loss_train_x3 = self.loss.loss_train_x3
        loss_test = self.loss.loss_test

        Fig=plt.figure(figsize=(10, 4))
        ax1=Fig.add_subplot(1,1,1)
        ax1.semilogy(self.loss.steps, loss_train,'--k', label="Training loss")
        ax1.semilogy(self.loss.steps, loss_train_x1,':',lw=2, label="$\mathcal{L}_{\mathbf{y}_1}$")
        ax1.semilogy(self.loss.steps, loss_train_bc, label="$\mathcal{L}_{\mathbf{BC}}$")
        ax1.semilogy(self.loss.steps, loss_train_x2,':',lw=2, label="$\mathcal{L}_{\mathbf{y}_2}$")
        #plt.semilogy(self.loss.steps, loss_train_f,'--k', label="ode")
        ax1.semilogy(self.loss.steps, loss_test,'-k',lw=2, label="Validation loss")


        ax1.semilogy(self.loss.steps, loss_train_x3,':',lw=2, label="$\mathcal{L}_{\mathbf{y}_3}$")
        ax1.grid()

        # for i in range(len(losshistory.metrics_test[0])):
        #     plt.semilogy(
        #         losshistory.steps,
        #         np.array(losshistory.metrics_test)[:, i],
        #         label="Test metric",
        #     )
        plt.setp(ax1.get_xticklabels(), fontsize=Font)
        plt.setp(ax1.get_yticklabels(), fontsize=Font)
        plt.xlabel("Epochs",fontsize=Font)
        plt.legend(bbox_to_anchor=(0.4, 1.0), ncol = 3,fontsize=Font)
        return Fig
        
    def __prep_data_plot(self,model):
        train_X,_, test_X , _=self.pack_plot
        
        y_pred_train=model.predict(train_X)
        y_pred_train=y_pred_train.reshape(y_pred_train.shape[0]*y_pred_train.shape[1],y_pred_train.shape[2])
        y_pred_test=model.predict(test_X)
        y_pred_test=y_pred_test.reshape(y_pred_test.shape[0]*y_pred_test.shape[1],y_pred_test.shape[2])
        n_steps_out=1
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
        return pred_train*self.xc+self.x0, pred_test*self.xc+self.x0
    def gen_plot_result(self):   
        if self.obs.shape[-1]==2:
            self.obs=self.obs*self.xc[0:2]+self.x0[0:2]
        if self.obs.shape[-1]==3:
            self.obs=self.obs*self.xc+self.x0
            #y_mean = np.mean(prediction1)
    
        k = np.arange(0,len(self.obs))
        ktr=np.arange(0,len(self.pred_train))
        kts=np.arange(len(self.pred_train),len(self.pred_train)+len(self.pred_test))
        #print(k.shape,ktr.shape,kts.shape)
        Fig=plt.figure(figsize=(10, 4))
        label=["$P_{bh}(bar)$","$P_{wh}(bar)$","$q (m^3/h)$"]
        scale=[1/1e5,1/1e5,3600]
        cor=["black","black","gray"]
        leg_lb=['Observed data','Observed data','No measured data']


        for i,val in enumerate(label):
            ax1=Fig.add_subplot(len(label),1,i+1)
            if i==2:
                l0, =ax1.plot(k, self.obs[:,i]*scale[i],"-",color=cor[i])
            else:
                l1, =ax1.plot(k, self.obs[:,i]*scale[i],"-",color=cor[i])

            l2, =ax1.plot(ktr, self.pred_train[:,i]*scale[i],":",color='red',lw=2)
            l3, =ax1.plot(kts, self.pred_test[:,i]*scale[i],":",color='blue',lw=2)
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
    
    