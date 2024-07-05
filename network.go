package main

import (
	"fmt"
	"math"
)

type NeuralNet struct {
	Weights []Matrix
	Biases  []Matrix
}

func newNeuralNet(neurons []int) *NeuralNet {
	weights := make([]Matrix, len(neurons)-1)
	biases := make([]Matrix, len(neurons)-1)
	for i := range weights {
		weights[i] = NewMatrix(neurons[i], neurons[i+1], "random")
		biases[i] = NewMatrix(1, neurons[i+1], "random")
	}
	return &NeuralNet{Weights: weights, Biases: biases}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func costDerivative(outp, y Matrix) Matrix {
	derivative := outp.Sub(y)
	return derivative
}

func computeZ(weights, activations, biases Matrix) Matrix {
	sampleSize := activations.Shape()[0]
	z := activations.MatMul(weights).Add(biases.VDuplicate(sampleSize))
	return z
}

func (nn *NeuralNet) Forward(input Matrix) Matrix {
	x := input
	for i := range nn.Weights {
		x = computeZ(nn.Weights[i], x, nn.Biases[i])
		x = x.Apply(sigmoid)
	}
	return x
}

func (nn *NeuralNet) evaluate(X, y []Matrix) float64 {
	correct := 0.0
	total := float64(len(X) * X[0].Shape()[0])
	for i := range X {
		output := nn.Forward(X[i])
		prediction := output.Argmax()
		labels := y[i].Argmax()
		for j := 0; j < labels.Shape()[0]; j++ {
			pred := prediction[j][0]
			lab := labels[j][0]
			if pred == lab {
				correct++
			}
		}
	}
	return correct / total
}

func (nn *NeuralNet) backprop(X, y Matrix) ([]Matrix, []Matrix) {
	nLayers := len(nn.Weights)
	nablaW := make([]Matrix, nLayers)
	nablaB := make([]Matrix, nLayers)
	activations := make([]Matrix, nLayers+1)
	zs := make([]Matrix, nLayers)
	activations[0] = X

	// forward
	for i := 0; i < nLayers; i++ {
		z := computeZ(nn.Weights[i], activations[i], nn.Biases[i])
		zs[i] = z
		nextActivation := z.Apply(sigmoid)
		activations[i+1] = nextActivation
	}

	//backward
	delta := costDerivative(activations[nLayers], y).Mul(zs[nLayers-1].Apply(sigmoidPrime))
	nablaB[nLayers-1] = delta.MeanOfSamples()
	nablaW[nLayers-1] = activations[nLayers-1].Transpose().MatMul(delta).DivScalar(float64(delta.Shape()[0]))

	for i := nLayers - 2; i >= 0; i-- {
		z := zs[i]
		delta = delta.MatMul(nn.Weights[i+1].Transpose()).Mul(z.Apply(sigmoidPrime))
		nablaB[i] = delta.MeanOfSamples()
		nablaW[i] = activations[i].Transpose().MatMul(delta).DivScalar(float64(delta.Shape()[0]))
	}

	return nablaW, nablaB
}

func (nn *NeuralNet) updateWeights(X, y Matrix, lr float64) {
	nablaW, nablaB := nn.backprop(X, y)
	for i := 0; i < len(nn.Weights); i++ {
		nn.Weights[i] = nn.Weights[i].Sub(nablaW[i].MulScalar(lr))
		nn.Biases[i] = nn.Biases[i].Sub(nablaB[i].MulScalar(lr))
	}
}

func (nn *NeuralNet) sgd(X_train_dl, y_train_dl, X_val_dl, y_val_dl []Matrix,
	epochs int, lr float64) {
	for i := 0; i < epochs; i++ {
		for j := 0; j < len(X_train_dl); j++ {
			nn.updateWeights(X_train_dl[j], y_train_dl[j], lr)
		}
		fmt.Printf("Epoch %v/%v\tAccuracy: %.2f%%\n", i+1, epochs, nn.evaluate(X_val_dl, y_val_dl)*100.0)
	}
}
