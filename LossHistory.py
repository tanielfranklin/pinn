import matplotlib.pyplot as plt
class LossHistory(object):
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.loss_train_bc=[]
        self.loss_train_f=[]
        self.loss_train_x1=[]
        self.loss_train_x2=[]
        self.loss_train_x3=[]
        self.metrics_test = []
        self.loss_weights = 1
        self.Font=14

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train,loss_train_bc,loss_train_f,loss_train_x1,loss_train_x2,loss_train_x3, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        self.loss_train_bc.append(loss_train_bc)
        self.loss_train_f.append(loss_train_f)
        self.loss_train_x1.append(loss_train_x1)
        self.loss_train_x2.append(loss_train_x2)
        self.loss_train_x3.append(loss_train_x3)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        # if metrics_test is None:
        #     metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        #self.metrics_test.append(metrics_test)

    def plot_loss_res(self):
        # loss_train = np.sum(losshistory.loss_train, axis=0)
        # loss_test = np.sum(losshistory.loss_test, axis=0)
        loss_train = self.loss_train
        loss_train_bc = self.loss_train_bc
        loss_train_f = self.loss_train_f
        loss_train_x1 = self.loss_train_x1
        loss_train_x2 = self.loss_train_x2
        loss_train_x3 = self.loss_train_x3
        loss_test = self.loss_test

        Fig=plt.figure(figsize=(10, 4))
        ax1=Fig.add_subplot(1,1,1)
        ax1.semilogy(self.steps, loss_train,'--k', label="Training loss")
        ax1.semilogy(self.steps, loss_train_x1,':',lw=2, label="$\mathcal{L}_{\mathbf{y}_1}$")
        ax1.semilogy(self.steps, loss_train_bc, label="$\mathcal{L}_{\mathbf{BC}}$")
        ax1.semilogy(self.steps, loss_train_x2,':',lw=2, label="$\mathcal{L}_{\mathbf{y}_2}$")
        #plt.semilogy(self.steps, loss_train_f,'--k', label="ode")
        ax1.semilogy(self.steps, loss_test,'-k',lw=2, label="Validation loss")


        ax1.semilogy(self.steps, loss_train_x3,':',lw=2, label="$\mathcal{L}_{\mathbf{y}_3}$")
        ax1.grid()

        # for i in range(len(losshistory.metrics_test[0])):
        #     plt.semilogy(
        #         losshistory.steps,
        #         np.array(losshistory.metrics_test)[:, i],
        #         label="Test metric",
        #     )
        plt.setp(ax1.get_xticklabels(), fontsize=Font)
        plt.setp(ax1.get_yticklabels(), fontsize=Font)
        plt.xlabel("Epochs",fontsize=self.Font)
        plt.legend(bbox_to_anchor=(0.9, 1.0), ncol = 3,fontsize=self.Font)
        return Fig
    
    def plot_loss(self):
        # loss_train = np.sum(losshistory.loss_train, axis=0)
        # loss_test = np.sum(losshistory.loss_test, axis=0)
        loss_train = self.loss_train
        # loss_train_bc = self.loss_train_bc
        # loss_train_f = self.loss_train_f
        # loss_train_x1 = self.loss_train_x1
        # loss_train_x2 = self.loss_train_x2
        # loss_train_x3 = self.loss_train_x3
        loss_test = self.loss_test

        plt.figure()
        plt.semilogy(self.steps, loss_train,':k', label="Train loss")
        # plt.semilogy(self.steps, loss_train_bc, label="bc")
        # plt.semilogy(self.steps, loss_train_f,'--k', label="ode")
        plt.semilogy(self.steps, loss_test,'-k', label="Test loss")
        # plt.semilogy(self.steps, loss_train_x1,':', label="x1")
        # plt.semilogy(self.steps, loss_train_x2,':', label="x2")
        # plt.semilogy(self.steps, loss_train_x3,':', label="x3")

        # for i in range(len(losshistory.metrics_test[0])):
        #     plt.semilogy(
        #         losshistory.steps,
        #         np.array(losshistory.metrics_test)[:, i],
        #         label="Test metric",
        #     )
        plt.xlabel("Epochs",fontsize=self.Font)
        #plt.legend(bbox_to_anchor=(0.9, -0.15), ncol = 1)
        plt.legend()

    def plotly_losses(self):
        loss_train = self.loss_train
        loss_train_bc = self.loss_train_bc
        loss_train_f = self.loss_train_f
        loss_train_x1 = self.loss_train_x1
        loss_train_x2 = self.loss_train_x2
        loss_train_x3 = self.loss_train_x3
        loss_test = self.loss_test

        df2 = pd.DataFrame(dict(
                Epochs = self.steps,
                Trainloss = loss_train,
                Testloss = loss_test,
                x1=loss_train_x1,
                x2=loss_train_x2,
                x3=loss_train_x3,
                ode=loss_train_f,
                bc=loss_train_bc
                ))
        fig = px.line(df2, x="Epochs", y=["Trainloss",'Testloss','x1','x2',"x3","ode","bc"] ,log_y=True)
        return fig



class VarHistory(object):
    def __init__(self):
        self.steps = []
        self.rho_train = []
        self.PI_train = []
        self.best_rho = 0
        self.best_PI = 0
        self.best_step = 0
        self.Font=14
    def append(self, step, rho_train, PI_train):
        self.steps.append(step)
        self.rho_train.append(rho_train)
        self.PI_train.append(PI_train)
    def update_best(self,trainstate):    
        self.best_step = trainstate.best_step
        self.best_rho = trainstate.rho.numpy()
        self.best_PI = trainstate.PI.numpy()

    def plot_var(self,lim_rho=None,lim_PI=None):
        rho_train = self.rho_train
        Font=self.Font
        PI_train = self.PI_train
        steps=self.steps
        rho_ref=950
        conv=3.6e8;
        PI_ref=2.32e-9*conv
        PI_train = [element * conv for element in PI_train]
        label=[r"Estimated $\rho$", 'Estimated PI', r"True $\rho$", "True PI"]

        #fig=plt.figure()
        fig=plt.figure(figsize=(7, 4))
        ax=fig.add_subplot()
        ln1=ax.plot(steps,rho_train,'-k', label=label[0])
        ln3=ax.plot(steps,np.ones((len(steps),1))*rho_ref,'--k',label=label[2])
        #ln5=plt.annotate("X", (int(self.best_step),self.best_rho))
        #ln5=ax.plot(varhistory.best_step,varhistory.best_rho,'k',marker='x',label=r'Best $\rho$')
        
        #ax.set(ylim=(800, 1200))
        #ax2.set(ylim=(0.72, 0.9 ))
        plt.setp(ax.get_xticklabels(), fontsize=Font)
        plt.setp(ax.get_yticklabels(), fontsize=Font)
        ax2=ax.twinx()

        ax2.set_ylabel(r'$PI~[m^3/h/bar]$',fontsize=Font)
        ln2=ax2.plot(steps,PI_train,'-',color='gray', label=label[1])
        ln4=ax2.plot(steps,np.ones_like(steps)*PI_ref,'--',color='gray', label=label[3])
        #plt.annotate("X", (int(self.best_step),self.best_PI*conv))
        #ln6=ax2.plot(varhistory.best_step,varhistory.best_PI*conv,'k',marker='x' ,label='Best PI')
        
        

        # if PI_lim!=None:
        #ax2.set(ylim=(0.72, 0.9 ))
        if lim_rho!=None:
            ax.set(ylim=(lim_rho[0], lim_rho[1]))
        if lim_PI!=None:    
            ax2.set(ylim=(lim_PI[0], lim_PI[1] ))
        # added these three lines
        # ln = ln1+ln2#+ln2+ln3
        
        # labs = [l.get_label() for l in ln]
        #ax2.legend(ln, labs, loc='lower right')#,ncol=2)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles2, labels2, loc='lower right',frameon=False)#,ncol=2)
        plt.setp(ax2.get_xticklabels(), fontsize=Font)
        plt.setp(ax2.get_yticklabels(), fontsize=Font)
        leg = Legend(ax2, handles, labels,loc='lower center', frameon=False)

        ax2.add_artist(leg)
        ax.set_ylabel(r"$\rho ~[kg/m^3]$",fontsize=Font)
        ax.set_xlabel('Epochs',fontsize=Font)
        plt.grid(True)
        #plt.show()
        return fig