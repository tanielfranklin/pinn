PINN VFM <br>
If this system help you, you are encouraged to cite the following paper:<br>

    ---
    ---
    References
    ==========

    @book{ruby,
      title     = {The Ruby Programming Language},
      author    = {Flanagan, David and Matsumoto, Yukihiro},
      year      = {2008},
      publisher = {O'Reilly Media}
    }

There are some datasets available. We build dataset01 (using main_build_dataset.py) for training purposes and dataset_opera (using main_build_dataset.py) to verify the model generability. <br>

The procedure to achieve a good model is: <br>
a) start the training using model_adam_200 (main_train_adam.py)  <br>
b) complete the training using changing to main_train_lbfgs resulting in model_adam_lbfgs model <br>