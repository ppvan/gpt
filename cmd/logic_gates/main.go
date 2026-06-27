package main

import (
	"fmt"

	"github.com/ppvan/gpt/pkg/nn"
)

func main() {
	xor()
	// nand()
}

func xor() {
	x := nn.NewMat([][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	})
	labels := nn.NewMat([][]float64{
		{0},
		{1},
		{1},
		{0},
	})

	data := nn.Data{
		X: x, Y: labels,
	}

	model := nn.NewSequential(
		nn.NewLinear(2, 4),
		nn.LeakyRelu(0.1),
		nn.NewLinear(4, 16),
		nn.LeakyRelu(0.1),
		nn.NewLinear(16, 2),
	)

	net := nn.NewNetwork(model, nn.CrossEntropy())

	for m := range net.Fit(data, 5000, 4) {
		fmt.Printf("epoch=%d loss=%.4f\r", m.Epoch, m.Loss)
	}
	fmt.Println()

	fmt.Println("===== FINAL PREDICTIONS =====")
	for i := 0; i < x.Rows; i++ {
		row := x.RowAt(i)
		input := nn.NewRowMat(row)

		class := net.Predict(input)
		fmt.Printf("%v | %v = %v\n", row[0], row[1], class.Get(0, 0))
	}
}
