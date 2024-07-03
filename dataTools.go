package main

import (
	t "gorgonia.org/tensor"
)

func dataLoader(X, y t.Tensor, batchSize int) ([]t.Tensor, []t.Tensor) {
	// no need to shuffle as dataset is already shuffled
	nBatches := X.Shape()[0] / batchSize // so we drop last
	X_dl, y_dl := make([]t.Tensor, nBatches), make([]t.Tensor, nBatches)
	for i := 0; i < nBatches; i++ {
		X_dl[i], _ = X.Slice(t.S(i*batchSize, (i+1)*batchSize), nil)
		y_dl[i], _ = y.Slice(t.S(i*batchSize, (i+1)*batchSize), nil)
	}
	return X_dl, y_dl
}

func meanOver(X t.Tensor, dim int) t.Tensor {
	n := float64(X.Shape()[dim])
	sum, _ := t.Sum(X, dim)
	mean, _ := t.Div(sum, n)
	return mean
}
