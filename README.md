# Neural Network in Golang
When i started this project I didn't even know how to write "Hello World!" in Go and here i am presenting the neural network

## Prototype in Python
It is essential to start with a prototype to design an optimal neural net architecture and to identify some potential flaws. That's exactly what i did. I created a [prototype](https://github.com/Mshkf/Neural-Network-in-Golang/blob/main/mnist-prototype.ipynb) in Jupyter Notebook and chose the following architecture (inspired by 3blue1brown videos):
* Fully connected layers [784,16,16,10] with a sigmoid activation
* MSE as a loss function
* SGD as an optimizer
  
I didn't use GPU calculations, ADAM optimizer or parallel computing in dataloaders since it would be even more laborious to code that in Go.
### Results
The code was running in Kaggle notebook.

With that architecture I managed to get 94,65% accuracy on validation data after 25 epochs and it took model 4 minutes to do so (remember that time as i will compare it later).

For the sake of experiment i also applied all the optimizations I mentioned above and it took 2 minutes for model to train
## Golang
### Dependecies
There are only 2 Go packages
* [tensor](https://pkg.go.dev/gorgonia.org/tensor) to work with tensors
* [mnist](https://pkg.go.dev/gorgonia.org/gorgonia/examples/mnist) to convert binary mnist data
### Results
The code was running on my laptop

With exactly the same parameters (including learning rate) i got 93% accuracy (it can be different due to different rules to initialize weights) and it took model 30 seconds to learn, which is quite impressive considering that my laptop is probably weaker than Kaggle's CPUs

So we got at least an 8-fold acceleration in computation time. Whether it was worth it or not remains an open question for the reader, but coding it was much more time-consuming (and a little bit more painful) than in python
