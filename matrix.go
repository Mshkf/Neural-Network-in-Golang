package main

import (
	"fmt"
	"math/rand"
	"strings"
)

type Matrix [][]float64

func NewMatrix(rows, cols int, filling string) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)

		switch filling {
		case "empty":
		case "zeros":
			// all values are 0 by default
			// but that is not explicit
		case "ones":
			for j := range m[i] {
				m[i][j] = 1
			}
		case "random":
			for j := range m[i] {
				m[i][j] = rand.NormFloat64()
			}
		default:
			panic("Unsupported filling\nAvailable fillings:\n" +
				"empty, zeros, ones, random")
		}
	}
	return m
}

func (m Matrix) Shape() []int {
	return []int{len(m), len(m[0])}
}

func (m Matrix) String() string {
	var b strings.Builder

	for _, row := range m {
		b.WriteString(fmt.Sprintln(row))
	}

	return b.String()
}

func (m1 Matrix) Add(m2 Matrix) Matrix {
	if len(m1) != len(m2) || len(m1[0]) != len(m2[0]) {
		panic("Matrices have different dimensions")
	}
	result := NewMatrix(len(m1), len(m1[0]), "empty")
	for i := range m1 {
		for j := range m1[0] {
			result[i][j] = m1[i][j] + m2[i][j]
		}
	}
	return result
}

func (m1 Matrix) Sub(m2 Matrix) Matrix {
	if len(m1) != len(m2) || len(m1[0]) != len(m2[0]) {
		panic("Matrices have different dimensions")
	}
	result := NewMatrix(len(m1), len(m1[0]), "empty")
	for i := range m1 {
		for j := range m1[0] {
			result[i][j] = m1[i][j] - m2[i][j]
		}
	}
	return result
}

func (m1 Matrix) Mul(m2 Matrix) Matrix {
	if len(m1) != len(m2) || len(m1[0]) != len(m2[0]) {
		panic("Matrices have different dimensions")
	}
	result := NewMatrix(len(m1), len(m1[0]), "empty")
	for i := range m1 {
		for j := range m1[0] {
			result[i][j] = m1[i][j] * m2[i][j]
		}
	}
	return result
}

func (m Matrix) MulScalar(scalar float64) Matrix {
	result := NewMatrix(len(m), len(m[0]), "empty")
	for i := range m {
		for j := range m[0] {
			result[i][j] = m[i][j] * scalar
		}
	}
	return result
}

func (m1 Matrix) Div(m2 Matrix) Matrix {
	if len(m1) != len(m2) || len(m1[0]) != len(m2[0]) {
		panic("Matrices have different dimensions")
	}
	result := NewMatrix(len(m1), len(m1[0]), "empty")
	for i := range m1 {
		for j := range m1[0] {
			result[i][j] = m1[i][j] / m2[i][j]
		}
	}
	return result
}

func (m Matrix) DivScalar(scalar float64) Matrix {
	result := NewMatrix(len(m), len(m[0]), "empty")
	for i := range m {
		for j := range m[0] {
			result[i][j] = m[i][j] / scalar
		}
	}
	return result
}

func (m Matrix) Transpose() Matrix {
	// it is not an optimal way,
	// but adding a transpose property would overcomplicate code
	result := NewMatrix(len(m[0]), len(m), "empty")
	for i := range m[0] {
		for j := range m {
			result[i][j] = m[j][i]
		}
	}
	return result
}

// non-optimal matmul
//func (m1 Matrix) MatMul(m2 Matrix) Matrix {
//	if len(m1[0]) != len(m2) {
//		panic("Wrong dimentions for matrix multiplication")
//	}
//	result := NewMatrix(len(m1), len(m2[0]), "zeros")
//	for i := range m1 {
//		for j := range m2[0] {
//			for k := 0; k < len(m1[0]); k++ {
//				result[i][j] += m1[i][k] * m2[k][j]
//			}
//		}
//	}
//	return result
//}

func (m Matrix) Apply(fn func(float64) float64) Matrix {
	result := NewMatrix(len(m), len(m[0]), "empty")
	for i := range result {
		for j := range result[i] {
			result[i][j] = fn(m[i][j])
		}
	}
	return result
}

func (m1 Matrix) Argmax() Matrix {
	// again, it would be better to create int matrix via generics
	// but I won't do that for the sake of simplicity
	result := NewMatrix(len(m1), 1, "empty")
	for i := range m1 {
		k := 0
		for j := range m1[i] {
			if m1[i][j] > m1[i][k] {
				k = j
			}
		}
		result[i][0] = float64(k)
	}
	return result
}

func (m Matrix) MeanOfSamples() Matrix {
	result := NewMatrix(1, len(m[0]), "empty")
	for j := range m[0] {
		k := 0.0
		for i := range m {
			k += m[i][j]
		}
		result[0][j] = k / float64(len(m))
	}
	return result
}

func (m Matrix) SliceInBatches(batchSize int) []Matrix {
	nBatches := len(m) / batchSize
	result := make([]Matrix, nBatches)
	for i := 0; i < nBatches; i++ {
		result[i] = NewMatrix(batchSize, len(m[0]), "empty")
		for j := range batchSize {
			for k := 0; k < len(m[0]); k++ {
				result[i][j][k] = m[i*batchSize+j][k]
			}
		}
	}
	return result
}
