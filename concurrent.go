package main

// optimized matmul
func (m1 Matrix) MatMul(m2 Matrix) Matrix {
	if len(m1[0]) != len(m2) {
		panic("Wrong dimentions for matrix multiplication")
	}
	result := make(Matrix, len(m1))
	rowChannel := make(chan RowResult, len(m1))

	for i := 0; i < len(m1); i++ {
		row := make([]float64, len(m2[0]))
		go computeRow(rowChannel, row, m1[i], m2, i)
	}

	for x := 0; x < len(m1); x++ {
		select {
		case rowResult := <-rowChannel:
			idx := rowResult.index
			result[idx] = rowResult.result
		}
	}
	return result
}

func computeRow(c chan RowResult, resultRow, rowA []float64, m2 Matrix, i int) {
	for i, v := range rowA {
		for j := 0; j < len(m2[0]); j++ {
			resultRow[j] += m2[i][j] * v
		}
	}
	c <- RowResult{result: resultRow, index: i}
}

type RowResult struct {
	result []float64
	index  int
}
