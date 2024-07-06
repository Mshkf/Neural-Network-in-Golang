# Neural Network in pure Golang
When I started this project I didn't even know how to write "Hello World!" in Go and here I am presenting the neural network.

This network is written completely from scratch, without any external libraries (including matrix operations and mnist data loading)

## Prototype in Python
It is essential to start with a prototype to design an optimal neural net architecture and to identify some potential flaws. That's exactly what I did. I created a [prototype](https://github.com/Mshkf/Neural-Network-in-Golang/blob/main/mnist-prototype.ipynb) in Jupyter Notebook and chose the following architecture (inspired by 3blue1brown videos):
* Fully connected layers [784,16,16,10] with a sigmoid activation
* MSE as a loss function
* SGD as an optimizer
  
I didn't use GPU calculations, ADAM optimizer or parallel computing in dataloaders since it would be even more laborious to code that in Go.
### Results
The code was running in Kaggle notebook.

With that architecture I managed to get 94,65% accuracy on validation data after 25 epochs and it took model 4 minutes to do so (remember that time as I will compare it later).

For the sake of experiment I also applied all the optimizations I mentioned above and it took 2 minutes for model to train
## Golang
Unfortunately, my implementation of matrix operations isn't as fast as [external](https://pkg.go.dev/gorgonia.org/tensor), but after I implemented concurrent computing for matrix multiplication I managed to get nearly the same results (see below)
### Results
The code was running on my laptop

With exactly the same parameters (including learning rate) I got 93% accuracy (it can be different due to different rules of initializing weights or due to the lack of additional shuffling).

When I used external package for matrix operations program ran in 30 seconds, which is quite impressive considering that my laptop is probably weaker than Kaggle's CPUs.

With my initial implementation the runtime was about 100 seconds, but reduced to 40 seconds with concurrent computing.

So we got at least an 8-fold acceleration in computation time. Whether it was worth it or not remains an open question for the reader, but coding it was much more time-consuming (and a little bit more painful) than in python
## Future plans
If I don't abandon this project I can improve it in following ways:
* Get rid of all dependencies and make the project run in pure Go (done :white_check_mark:)
* Optimize tensor operations (e.g. matrix multiplication) using concurrent computations (done :white_check_mark:)
* Add other activation functions and losses (ReLU and CrossEntropy being the most widespread)
* Add ADAM optimizer

Althougn I won't do it in the nearest future as I managed to get a job while working on this project :relaxed:
