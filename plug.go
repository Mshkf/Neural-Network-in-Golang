package main

import t "gorgonia.org/tensor"

func ToMatrix(tensor t.Tensor) Matrix {
	n, m := tensor.Shape()[0], tensor.Shape()[1]
	matrix := NewMatrix(n, m, "empty")
	data := tensor.Data().([]float64)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			matrix[i][j] = data[i*m+j]
		}
	}
	return matrix
}
