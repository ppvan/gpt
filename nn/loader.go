package nn

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
)

type Dataset struct {
	X Mat
	Y Mat
}

func (d Dataset) NumSamples() int {
	return d.X.Rows
}

func NewDataset(x, y Mat) Dataset {
	if x.Rows != y.Rows {
		panic(fmt.Sprintf("dataset: X has %d rows but Y has %d rows", x.Rows, y.Rows))
	}
	return Dataset{X: x, Y: y}
}

func LoadCSV(path string, outSize int, skipHeader bool) (Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return Dataset{}, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	r := csv.NewReader(bufio.NewReader(f))
	rows, err := r.ReadAll()
	if err != nil {
		return Dataset{}, fmt.Errorf("parse csv %s: %w", path, err)
	}
	if skipHeader && len(rows) > 0 {
		rows = rows[1:]
	}
	if len(rows) == 0 {
		return Dataset{}, fmt.Errorf("no data rows in %s", path)
	}

	cols := len(rows[0])
	featSize := cols - outSize
	if featSize <= 0 {
		return Dataset{}, fmt.Errorf("outSize %d >= column count %d", outSize, cols)
	}

	xData := make([][]float64, len(rows))
	yData := make([][]float64, len(rows))
	for i, row := range rows {
		if len(row) != cols {
			return Dataset{}, fmt.Errorf("row %d: expected %d columns, got %d", i, cols, len(row))
		}
		xData[i] = make([]float64, featSize)
		for j := 0; j < featSize; j++ {
			v, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return Dataset{}, fmt.Errorf("row %d col %d: %w", i, j, err)
			}
			xData[i][j] = v
		}
		yData[i] = make([]float64, outSize)
		for j := 0; j < outSize; j++ {
			v, err := strconv.ParseFloat(row[featSize+j], 64)
			if err != nil {
				return Dataset{}, fmt.Errorf("row %d target %d: %w", i, j, err)
			}
			yData[i][j] = v
		}
	}

	return NewDataset(NewMat(xData), NewMat(yData)), nil
}

type Loader struct {
	data      Dataset
	batchSize int
	order     []int // current sample order (identity or shuffled)
}

func NewLoader(data Dataset, batchSize int) *Loader {
	if batchSize <= 0 {
		batchSize = data.NumSamples()
	}
	order := make([]int, data.NumSamples())
	for i := range order {
		order[i] = i
	}
	return &Loader{data: data, batchSize: batchSize, order: order}
}

// Shuffle randomizes sample order in place (Fisher-Yates).
func (l *Loader) Shuffle() {
	rand.Shuffle(len(l.order), func(i, j int) {
		l.order[i], l.order[j] = l.order[j], l.order[i]
	})
}

func (l *Loader) Batches() []Dataset {
	n := len(l.order)
	batches := make([]Dataset, 0, (n+l.batchSize-1)/l.batchSize)
	for start := 0; start < n; start += l.batchSize {
		end := start + l.batchSize
		if end > n {
			end = n
		}
		idx := l.order[start:end]
		xRows := make([][]float64, len(idx))
		yRows := make([][]float64, len(idx))
		for i, sampleIdx := range idx {
			xRows[i] = l.data.X.RowAt(sampleIdx)
			yRows[i] = l.data.Y.RowAt(sampleIdx)
		}
		batches = append(batches, NewDataset(NewMat(xRows), NewMat(yRows)))
	}
	return batches
}

func (l *Loader) NewEpoch() []Dataset {
	l.Shuffle()
	return l.Batches()
}
