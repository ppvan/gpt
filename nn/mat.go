package nn

import (
	"fmt"
	"strconv"
	"strings"
)

type Mat struct {
	Weights [][]float64
	Row     int
	Column  int
}

func (mat Mat) Combine(other Mat, f func(a, b float64) float64) Mat {
	if mat.Row != other.Row || mat.Column != other.Column {
		message := fmt.Sprintf("In-compatible matrix, can't combine (%v x %v) * (%v x %v)", mat.Row, mat.Column, other.Row, other.Column)
		panic(message)
	}
	result := make([][]float64, mat.Row)
	for i := range mat.Weights {
		result[i] = make([]float64, len(mat.Weights[i]))
		for j := range mat.Weights[i] {
			result[i][j] = f(mat.Weights[i][j], other.Weights[i][j])
		}
	}
	return Mat{
		Weights: result,
		Row:     mat.Row,
		Column:  mat.Column,
	}
}

func (mat Mat) Add(other Mat) Mat {
	return mat.Combine(other, func(a, b float64) float64 { return a + b })
}

func (mat Mat) Subtract(other Mat) Mat {
	return mat.Combine(other, func(a, b float64) float64 { return a - b })
}

func (mat Mat) Hadamard(other Mat) Mat {
	return mat.Combine(other, func(a, b float64) float64 { return a * b })
}

func (mat Mat) Multiply(other Mat) Mat {
	if mat.Column != other.Row {
		message := fmt.Sprintf("In-compatible matrix, can't multiply (%v x %v) * (%v x %v)", mat.Row, mat.Column, other.Row, other.Column)
		panic(message)
	}
	result := make([][]float64, mat.Row)
	for i := 0; i < mat.Row; i++ {
		result[i] = make([]float64, other.Column)
		for j := 0; j < other.Column; j++ {
			sum := 0.0
			for k := 0; k < mat.Column; k++ {
				sum += mat.Weights[i][k] * other.Weights[k][j]
			}
			result[i][j] = sum
		}
	}
	return Mat{
		Weights: result,
		Row:     mat.Row,
		Column:  other.Column,
	}
}

func (mat Mat) Transpose() Mat {
	result := make([][]float64, mat.Column)
	for i := range result {
		result[i] = make([]float64, mat.Row)
		for j := range result[i] {
			result[i][j] = mat.Weights[j][i]
		}
	}
	return Mat{
		Weights: result,
		Row:     mat.Column,
		Column:  mat.Row,
	}
}

func (mat Mat) Apply(fn func(float64) float64) Mat {
	result := make([][]float64, mat.Row)
	for i := range mat.Weights {
		result[i] = make([]float64, len(mat.Weights[i]))
		for j := range mat.Weights[i] {
			result[i][j] = fn(mat.Weights[i][j])
		}
	}
	return Mat{
		Weights: result,
		Row:     mat.Row,
		Column:  mat.Column,
	}
}

func NewRowMat(data []float64) Mat {
	return Mat{
		Weights: [][]float64{data},
		Row:     1,
		Column:  len(data),
	}
}

func NewZeroMat(row, column int) Mat {
	result := make([][]float64, row)
	for i := range result {
		result[i] = make([]float64, column)
	}
	return Mat{
		Weights: result,
		Row:     row,
		Column:  column,
	}
}

func (mat Mat) String() string {
	var sb strings.Builder
	for i := range mat.Weights {
		sb.WriteString("[")
		for j, v := range mat.Weights[i] {
			if j > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(strconv.FormatFloat(v, 'f', 2, 64))
		}
		sb.WriteString("]")
		if i < len(mat.Weights)-1 {
			sb.WriteString("\n")
		}
	}
	return sb.String()
}
