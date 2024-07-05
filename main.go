package main

import (
	"fmt"
	"gorgonia.org/gorgonia/examples/mnist"
	t "gorgonia.org/tensor"
	"time"
)

func main() {
	X_train, y_train, err := mnist.Load("train", "./data", t.Float64)
	checkData(X_train, y_train, err, "train")
	X_val, y_val, err := mnist.Load("test", "./data", t.Float64)
	checkData(X_val, y_val, err, "test")

	X_train_m, y_train_m := ToMatrix(X_train), ToMatrix(y_train)
	X_val_m, y_val_m := ToMatrix(X_val), ToMatrix(y_val)

	X_train_dl, y_train_dl := X_train_m.SliceInBatches(64), y_train_m.SliceInBatches(64)
	X_val_dl, y_val_dl := X_val_m.SliceInBatches(64), y_val_m.SliceInBatches(64)

	saveImage(X_train_dl[1])

	NN := newNeuralNet([]int{784, 16, 16, 10})

	start := time.Now()
	NN.sgd(X_train_dl, y_train_dl, X_val_dl, y_val_dl, 25, 5.0)
	fmt.Printf("Training took %s", time.Since(start))

}
