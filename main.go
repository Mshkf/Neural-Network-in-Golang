package main

import (
	"gorgonia.org/gorgonia/examples/mnist"
	t "gorgonia.org/tensor"
)

func main() {
	X_train, y_train, err := mnist.Load("train", "./data", t.Float64)
	checkData(X_train, y_train, err, "train")
	X_val, y_val, err := mnist.Load("test", "./data", t.Float64)
	checkData(X_val, y_val, err, "test")

	X_train_dl, y_train_dl := dataLoader(X_train, y_train, 64)
	X_val_dl, y_val_dl := dataLoader(X_val, y_val, 64)

	saveImage(X_train_dl[1])

	NN := newNeuralNet([]int{784, 16, 16, 10})

	NN.sgd(X_train_dl, y_train_dl, X_val_dl, y_val_dl, 25, 5.0)

}
