PINN VFM <br>
<!-- If this system help you, you are encouraged to cite the following paper:<br>

    ---
    ---
  @article{FRANKLIN2022,
title = {A Physics-Informed Neural Networks (PINN) oriented approach to flow metering in oil wells: an ESP lifted oil well system as a case study},
journal = {Digital Chemical Engineering},
pages = {100056},
year = {2022},
issn = {2772-5081},
doi = {https://doi.org/10.1016/j.dche.2022.100056},
url = {https://www.sciencedirect.com/science/article/pii/S2772508122000461},
author = {Taniel S. Franklin and Leonardo S. Souza and Raony M. Fontes and Márcio A.F. Martins},
keywords = {soft sensor, Physics-Informed Neural Networks, electrical submersible pump, virtual flow meter, recurrent neural network},

}-->
There are some datasets available. We build dataset01 (using main_build_dataset.py) for training purposes and dataset_opera (using main_build_dataset.py) to verify the model generability. <br>

The procedure to achieve a good model is: <br>
a) start the training using model_adam_200 (main_train_adam.py)  <br>
b) complete the training changing to main_train_lbfgs resulting in model_adam_lbfgs model <br>
c) pay attention to the loss terms evolution to set a better set of weights <br>

The notebook main.ipynb demonstrates the model capability during free prediction <br>
<br>
Feel free to contribute with the software improvements.

