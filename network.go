package main

import (
	"fmt"
	t "gorgonia.org/tensor"
	"math"
)

type NeuralNet struct {
	Weights []t.Tensor
	Biases  []t.Tensor
}

func newNeuralNet(neurons []int) *NeuralNet {
	weights := make([]t.Tensor, len(neurons)-1)
	biases := make([]t.Tensor, len(neurons)-1)
	for i := range weights {
		weights[i] = t.New(t.WithShape(neurons[i], neurons[i+1]),
			t.WithBacking(t.Random(t.Float64, (neurons[i])*neurons[i+1])))
		biases[i] = t.New(t.WithShape(1, neurons[i+1]),
			t.WithBacking(t.Random(t.Float64, neurons[i+1])))
	}
	return &NeuralNet{Weights: weights, Biases: biases}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func costDerivative(outp, y t.Tensor) t.Tensor {
	derivative, _ := t.Sub(outp, y)
	return derivative
}

func computeZ(weights, activations, biases t.Tensor) t.Tensor {
	wPart, _ := t.Dot(activations, weights)
	ones := t.Ones(t.Float64, activations.Shape()[0], 1)
	bPart, _ := t.MatMul(ones, biases)
	z, _ := t.Add(wPart, bPart)
	return z
}

func (nn *NeuralNet) Forward(input t.Tensor) t.Tensor {
	x := input
	for i := range nn.Weights {
		x = computeZ(nn.Weights[i], x, nn.Biases[i])
		x, _ = x.Apply(sigmoid)
	}
	return x
}

func (nn *NeuralNet) evaluate(X, y []t.Tensor) float64 {
	correct := 0.0
	total := float64(len(X) * X[0].Shape()[0])
	for i := range X {
		output := nn.Forward(X[i])
		prediction, _ := t.Argmax(output, 1)
		labels, _ := t.Argmax(y[i], 1)
		for j := 0; j < labels.Shape()[0]; j++ {
			pred, _ := prediction.At(j)
			lab, _ := labels.At(j)
			if pred == lab {
				correct++
			}
		}
	}
	return correct / total
}

func (nn *NeuralNet) backprop(X, y t.Tensor) ([]t.Tensor, []t.Tensor) {
	nLayers := len(nn.Weights)
	nablaW := make([]t.Tensor, nLayers)
	nablaB := make([]t.Tensor, nLayers)
	activations := make([]t.Tensor, nLayers+1)
	zs := make([]t.Tensor, nLayers)
	activations[0] = X

	// forward
	for i := 0; i < nLayers; i++ {
		z := computeZ(nn.Weights[i], activations[i], nn.Biases[i])
		zs[i] = z
		nextActivation, _ := z.Apply(sigmoid)
		activations[i+1] = nextActivation
	}

	//backward
	delta, _ := t.Mul(costDerivative(activations[nLayers], y),
		noErr(zs[nLayers-1].Apply(sigmoidPrime)))
	nablaB[nLayers-1] = meanOver(delta, 0)
	nablaW[nLayers-1], _ = t.Div(noErr(t.MatMul(noErr(t.T(activations[nLayers-1])), delta)),
		float64(delta.Shape()[0]))

	for i := nLayers - 2; i >= 0; i-- {
		z := zs[i]
		delta, _ = t.Mul(noErr(t.MatMul(delta, noErr(t.T(nn.Weights[i+1])))), noErr(z.Apply(sigmoidPrime)))
		nablaB[i] = meanOver(delta, 0)
		nablaW[i], _ = t.Div(noErr(t.MatMul(noErr(t.T(activations[i])), delta)),
			float64(delta.Shape()[0]))
	}

	return nablaW, nablaB
}

func (nn *NeuralNet) updateWeights(X, y t.Tensor, lr float64) {
	nablaW, nablaB := nn.backprop(X, y)
	for i := 0; i < len(nn.Weights); i++ {
		nn.Weights[i], _ = t.Sub(nn.Weights[i], noErr(t.Mul(nablaW[i], lr)))
		nn.Biases[i], _ = t.Sub(nn.Biases[i], noErr(t.Mul(nablaB[i], lr)))
	}
}

func (nn *NeuralNet) sgd(X_train_dl, y_train_dl, X_val_dl, y_val_dl []t.Tensor,
	epochs int, lr float64) {
	for i := 0; i < epochs; i++ {
		for j := 0; j < len(X_train_dl); j++ {
			nn.updateWeights(X_train_dl[j], y_train_dl[j], lr)
		}
		fmt.Printf("Epoch %v/%v\tAccuracy: %.2f%%\n", i+1, epochs, nn.evaluate(X_val_dl, y_val_dl)*100.0)
	}
}
