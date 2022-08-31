PINN VFM <br>
If this system help you, you are encouraged to cite the following paper:<br>

    ---
    ---
    @article{franklin2021,
      title     = {A Physics-Informed Neural Networks (PINN) oriented approach to flow
metering in oil wells: an ESP lifted oil well system as a case study},
      author    = { Franklin,Taniel S.  and  Souza, Leonardo S. and  Fontes, Raony M. and  Martins, Márcio A. F.},
      year      = {2022},
      volume={5},
      journal = {Digital Chemical Engineering}
    }

There are some datasets available. We build dataset01 (using main_build_dataset.py) for training purposes and dataset_opera (using main_build_dataset.py) to verify the model generability. <br>

The procedure to achieve a good model is: <br>
a) start the training using model_adam_200 (main_train_adam.py)  <br>
b) complete the training using changing to main_train_lbfgs resulting in model_adam_lbfgs model <br>