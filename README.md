# federated-average-implementation
Implementation of Federated Average in Federated Learning Environment

The custom implementation is found in `main.ipynb` jupyter notebook.

The tensorflow federated simulation results can be found in `main.py`

> Use a python virtual envronment with python 3.10 for an error less experience

## Working

The `main.ipynb` file contains details about the server and client code which exchange models by exporting them at each step.

The `main.py` file contains official exmaple implementation by TensorFlow Federated (taken from the official website)

## Results

Both the `main.ipynb` and `main.py` files try to use federated learning to train a model on the classical MNIST dataset spread between clients.

Final server aggregate models are found to have ~10% accuracy. Thus, the federated average implementation can be deemed correct.

## Contact

If you find any problems and suggestion over this, please raise a issue or contact me.