package main

import (
	"fmt"
	"time"
)

func main() {
	X_train := readImageFile("./data/train-images-idx3-ubyte")
	y_train := readLabelFile("./data/train-labels-idx1-ubyte")
	X_val := readImageFile("./data/t10k-images-idx3-ubyte")
	y_val := readLabelFile("./data/t10k-labels-idx1-ubyte")

	X_train_dl, y_train_dl := X_train.SliceInBatches(64), y_train.SliceInBatches(64)
	X_val_dl, y_val_dl := X_val.SliceInBatches(64), y_val.SliceInBatches(64)

	saveImage(X_train_dl[1])

	NN := newNeuralNet([]int{784, 16, 16, 10})

	start := time.Now()
	NN.sgd(X_train_dl, y_train_dl, X_val_dl, y_val_dl, 25, 5.0)
	fmt.Printf("Training took %s", time.Since(start))

}
