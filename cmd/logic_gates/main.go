package main

import (
	"context"
	"fmt"

	"github.com/ppvan/gpt/pkg/mm"
)

func main() {
	xor()
	// nand()
}

func xor() {
	x := mm.NewMat([][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	})
	labels := mm.NewMat([][]float64{
		{0},
		{1},
		{1},
		{0},
	})
	data := mm.Data{
		X: x, Y: labels,
	}
	model := mm.NewMultiLayerPerceptron(
		mm.NewLinear(2, 4),
		mm.NewLeakyReLU(0.1),
		mm.NewLinear(4, 16),
		mm.NewLeakyReLU(0.1),
		mm.NewLinear(16, 2),
	)
	opt := mm.NewGradient(0.01)
	net := mm.NewNetwork(model, mm.CrossEntropy(), opt)

	ctx := context.Background()
	for m := range net.Fit(ctx, data, 5000, 4) {
		fmt.Printf("epoch=%d loss=%.4f\r", m.Epoch, m.Loss)
	}
	fmt.Println()

	fmt.Println("===== FINAL PREDICTIONS =====")
	for i := 0; i < x.Rows; i++ {
		row := x.RowAt(i)
		input := mm.NewRowMat(row)
		pred, err := net.Predict(input)
		if err != nil {
			fmt.Printf("%v | %v = error: %v\n", row[0], row[1], err)
			continue
		}
		fmt.Printf("%v | %v = %d\n", row[0], row[1], pred.Class)
	}
}
