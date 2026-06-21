package nn

import (
	"fmt"
	"strconv"
	"strings"
)

type Mat struct {
	weights []float64
	Row     int
	Column  int
}

// at returns the flat index for element (r, c).
func (mat Mat) at(r, c int) int {
	return r*mat.Column + c
}

// Get returns the element at row r, column c.
func (mat Mat) Get(r, c int) float64 {
	return mat.weights[mat.at(r, c)]
}

// Set assigns the element at row r, column c.
func (mat Mat) Set(r, c int, v float64) {
	mat.weights[mat.at(r, c)] = v
}

func (mat Mat) Combine(other Mat, f func(a, b float64) float64) Mat {
	if mat.Row != other.Row || mat.Column != other.Column {
		message := fmt.Sprintf("In-compatible matrix, can't combine (%v x %v) * (%v x %v)", mat.Row, mat.Column, other.Row, other.Column)
		panic(message)
	}
	result := make([]float64, len(mat.weights))
	for i := range mat.weights {
		result[i] = f(mat.weights[i], other.weights[i])
	}
	return Mat{
		weights: result,
		Row:     mat.Row,
		Column:  mat.Column,
	}
}

func (mat Mat) Add(other Mat) Mat {
	return mat.Combine(other, func(a, b float64) float64 { return a + b })
}

func (mat Mat) Sub(other Mat) Mat {
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
	result := make([]float64, mat.Row*other.Column)
	for i := 0; i < mat.Row; i++ {
		for k := 0; k < mat.Column; k++ {
			aik := mat.weights[i*mat.Column+k]
			if aik == 0 {
				continue // cheap skip; harmless if dense, helps if sparse-ish
			}
			rowOffset := i * other.Column
			otherOffset := k * other.Column
			for j := 0; j < other.Column; j++ {
				result[rowOffset+j] += aik * other.weights[otherOffset+j]
			}
		}
	}
	return Mat{
		weights: result,
		Row:     mat.Row,
		Column:  other.Column,
	}
}

func (mat Mat) Transpose() Mat {
	result := make([]float64, len(mat.weights))
	for i := 0; i < mat.Row; i++ {
		for j := 0; j < mat.Column; j++ {
			// (j, i) in the transposed (Column x Row) matrix
			result[j*mat.Row+i] = mat.weights[i*mat.Column+j]
		}
	}
	return Mat{
		weights: result,
		Row:     mat.Column,
		Column:  mat.Row,
	}
}

func (mat Mat) Apply(fn func(float64) float64) Mat {
	result := make([]float64, len(mat.weights))
	for i, v := range mat.weights {
		result[i] = fn(v)
	}
	return Mat{
		weights: result,
		Row:     mat.Row,
		Column:  mat.Column,
	}
}

func (mat Mat) Scale(s float64) Mat {
	return mat.Apply(func(v float64) float64 { return v * s })
}

func (mat Mat) Sum() float64 {
	var total float64
	for _, v := range mat.weights {
		total += v
	}
	return total
}

func (mat Mat) Count() int {
	return len(mat.weights)
}

func (mat Mat) Mean() float64 {
	return mat.Sum() / float64(mat.Count())
}

func (mat Mat) RowAt(r int) []float64 {
	out := make([]float64, mat.Column)
	copy(out, mat.weights[r*mat.Column:(r+1)*mat.Column])
	return out
}

func NewRowMat(data []float64) Mat {
	w := make([]float64, len(data))
	copy(w, data)
	return Mat{
		weights: w,
		Row:     1,
		Column:  len(data),
	}
}

func NewZeroMat(row, column int) Mat {
	return Mat{
		weights: make([]float64, row*column),
		Row:     row,
		Column:  column,
	}
}

func NewMat(data [][]float64) Mat {
	if len(data) == 0 {
		return Mat{Row: 0, Column: 0}
	}
	row := len(data)
	col := len(data[0])
	flat := make([]float64, 0, row*col)
	for _, r := range data {
		if len(r) != col {
			panic("NewMat: ragged input, all rows must have the same length")
		}
		flat = append(flat, r...)
	}
	return Mat{weights: flat, Row: row, Column: col}
}

func (mat Mat) String() string {
	var sb strings.Builder
	for i := 0; i < mat.Row; i++ {
		sb.WriteString("[")
		for j := 0; j < mat.Column; j++ {
			if j > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(strconv.FormatFloat(mat.Get(i, j), 'f', 2, 64))
		}
		sb.WriteString("]")
		if i < mat.Row-1 {
			sb.WriteString("\n")
		}
	}
	return sb.String()
}
