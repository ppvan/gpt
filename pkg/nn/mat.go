package nn

import (
	"fmt"
	"math"
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
		msg := fmt.Sprintf("incompatible matrices: (%v x %v) and (%v x %v)", mat.Rows, mat.Columns, other.Rows, other.Columns)
		panic(msg)
	}

	m, n, p := mat.Rows, mat.Columns, other.Columns

	result := make([]float64, m*p)

	a := mat.weights
	b := other.weights

	for i := range m {
		aRow := i * n
		cRow := i * p

		for k := range n {
			aik := a[aRow+k]
			bRow := k * p
			// slice onces so compiler remove bound checks
			c := result[cRow : cRow+p]
			bb := b[bRow : bRow+p]

			// manual loop unrolling
			j := 0
			for ; j+3 < p; j += 4 {
				c[j+0] += aik * bb[j+0]
				c[j+1] += aik * bb[j+1]
				c[j+2] += aik * bb[j+2]
				c[j+3] += aik * bb[j+3]
			}

			for ; j < p; j++ {
				c[j] += aik * bb[j]
			}
		}
	}

	return Mat{
		weights: result,
		Rows:    m,
		Columns: p,
	}
}

func (mat Mat) Transpose() Mat {
	result := make([]float64, len(mat.weights))

	rows, cols := mat.Rows, mat.Columns
	src := mat.weights

	for i := range rows {
		rowOffset := i * cols
		for j := range cols {
			result[j*rows+i] = src[rowOffset+j]
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

func randomUniform(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func XavierMat(row, column int) Mat {
	fanIn := float64(row)
	fanOut := float64(column)

	a := math.Sqrt(6.0 / (fanIn + fanOut))

	return NewZeroMat(row, column).Apply(func(f float64) float64 {
		return randomUniform(-a, a)
	})
}

func HeMat(row, column int) Mat {
	fanIn := float64(row)

	a := math.Sqrt(2.0 / fanIn)

	return NewZeroMat(row, column).Apply(func(f float64) float64 {
		return randomUniform(-a, a)
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
	if start < 0 || end > mat.Rows || start > end {
		panic("invalid slice range")
	}

	rows := end - start
	result := NewZeroMat(rows, mat.Columns)

	copy(
		result.weights,
		mat.weights[start*mat.Columns:end*mat.Columns],
	)

	return result
}

func (mat Mat) Row(index int) Mat {
	if index < 0 || index > mat.Rows-1 {
		panic("invalid row")
	}

	rows := 1
	result := NewZeroMat(rows, mat.Columns)

	copy(
		result.weights,
		mat.weights[index*mat.Columns:(index+1)*mat.Columns],
	)

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

func AppendRows(a, b Mat) Mat {
	if a.Rows == 0 {
		return b
	}
	result := NewZeroMat(a.Rows+b.Rows, a.Columns)
	for i := range a.Rows {
		for j := range a.Columns {
			result.Set(i, j, a.Get(i, j))
		}
	}
	for i := range b.Rows {
		for j := range b.Columns {
			result.Set(a.Rows+i, j, b.Get(i, j))
		}
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
