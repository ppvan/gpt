package nn

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"
)

type Mat struct {
	weights []float64
	Rows    int
	Columns int
}

// at returns the flat index for element (r, c).
func (mat Mat) at(r, c int) int {
	return r*mat.Columns + c
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
	if mat.Rows != other.Rows || mat.Columns != other.Columns {
		message := fmt.Sprintf("In-compatible matrix, can't combine (%v x %v) * (%v x %v)", mat.Rows, mat.Columns, other.Rows, other.Columns)
		panic(message)
	}
	result := make([]float64, len(mat.weights))
	for i := range mat.weights {
		result[i] = f(mat.weights[i], other.weights[i])
	}
	return Mat{
		weights: result,
		Rows:    mat.Rows,
		Columns: mat.Columns,
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
	if mat.Columns != other.Rows {
		message := fmt.Sprintf("In-compatible matrix, can't multiply (%v x %v) * (%v x %v)", mat.Rows, mat.Columns, other.Rows, other.Columns)
		panic(message)
	}
	result := make([]float64, mat.Rows*other.Columns)
	for i := 0; i < mat.Rows; i++ {
		for k := 0; k < mat.Columns; k++ {
			aik := mat.weights[i*mat.Columns+k]
			if aik == 0 {
				continue // cheap skip; harmless if dense, helps if sparse-ish
			}
			rowOffset := i * other.Columns
			otherOffset := k * other.Columns
			for j := 0; j < other.Columns; j++ {
				result[rowOffset+j] += aik * other.weights[otherOffset+j]
			}
		}
	}
	return Mat{
		weights: result,
		Rows:    mat.Rows,
		Columns: other.Columns,
	}
}

func (mat Mat) Transpose() Mat {
	result := make([]float64, len(mat.weights))
	for i := 0; i < mat.Rows; i++ {
		for j := 0; j < mat.Columns; j++ {
			// (j, i) in the transposed (Column x Row) matrix
			result[j*mat.Rows+i] = mat.weights[i*mat.Columns+j]
		}
	}
	return Mat{
		weights: result,
		Rows:    mat.Columns,
		Columns: mat.Rows,
	}
}

func (mat Mat) Apply(fn func(float64) float64) Mat {
	result := make([]float64, len(mat.weights))
	for i, v := range mat.weights {
		result[i] = fn(v)
	}
	return Mat{
		weights: result,
		Rows:    mat.Rows,
		Columns: mat.Columns,
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
	out := make([]float64, mat.Columns)
	copy(out, mat.weights[r*mat.Columns:(r+1)*mat.Columns])
	return out
}

func NewRowMat(data []float64) Mat {
	w := make([]float64, len(data))
	copy(w, data)
	return Mat{
		weights: w,
		Rows:    1,
		Columns: len(data),
	}
}

func NewZeroMat(row, column int) Mat {
	return Mat{
		weights: make([]float64, row*column),
		Rows:    row,
		Columns: column,
	}
}

func randomMat(row, column int) Mat {
	return NewZeroMat(row, column).Apply(func(f float64) float64 {
		return rand.Float64()*2 - 1
	})
}

func NewMat(data [][]float64) Mat {
	if len(data) == 0 {
		return Mat{Rows: 0, Columns: 0}
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
	return Mat{weights: flat, Rows: row, Columns: col}
}

// OneHot converts a single-column label matrix (Row x 1, containing
// class indices like 0..9) into a one-hot encoded matrix
// (Row x numClasses). Each row becomes all zeros except a 1 at the
// column matching that sample's label.
//
// Panics if mat is not a single column, or if any label is out of
// range [0, numClasses).
func (mat Mat) OneHot(numClasses int) Mat {
	if mat.Columns != 1 {
		panic(fmt.Sprintf("OneHot: expected a single-column matrix, got %d columns", mat.Columns))
	}
	result := NewZeroMat(mat.Rows, numClasses)
	for i := 0; i < mat.Rows; i++ {
		label := int(mat.Get(i, 0))
		if label < 0 || label >= numClasses {
			panic(fmt.Sprintf("OneHot: label %d at row %d out of range [0, %d)", label, i, numClasses))
		}
		result.Set(i, label, 1.0)
	}
	return result
}

func (mat Mat) Slice(start, end int) Mat {
	result := NewZeroMat(end-start, mat.Columns)

	for r := 0; r < mat.Rows; r++ {
		for c := 1; c < mat.Columns; c++ {
			result.Set(r, c, mat.Get(r, c))
		}
	}

	return result
}

func (mat Mat) ArgMax() Mat {
	if mat.Columns == 0 {
		panic("ArgMax: matrix has no columns")
	}

	result := NewZeroMat(mat.Rows, 1)

	for r := 0; r < mat.Rows; r++ {
		maxIdx := 0
		maxVal := mat.Get(r, 0)

		for c := 1; c < mat.Columns; c++ {
			v := mat.Get(r, c)
			if v > maxVal {
				maxVal = v
				maxIdx = c
			}
		}

		result.Set(r, 0, float64(maxIdx))
	}

	return result
}

func (mat Mat) String() string {
	var sb strings.Builder
	for i := 0; i < mat.Rows; i++ {
		sb.WriteString("[")
		for j := 0; j < mat.Columns; j++ {
			if j > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(strconv.FormatFloat(mat.Get(i, j), 'f', 2, 64))
		}
		sb.WriteString("]")
		if i < mat.Rows-1 {
			sb.WriteString("\n")
		}
	}
	return sb.String()
}
