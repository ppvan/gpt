package main

import (
	"fmt"

	"github.com/ppvan/gpt/nn"
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
	y := labels.OneHot(2)

	data := nn.Data{
		X: x, Y: y,
	}

	model := nn.NewSequential(
		nn.NewLinear(2, 4),
		nn.Sigmoid(),
		nn.NewLinear(4, 4),
		nn.Sigmoid(),
		nn.NewLinear(4, 2),
	)

	net := nn.NewNetwork(model, nn.CrossEntropy())

	for m := range net.Fit(data, 10240, 32) {
		fmt.Printf("epoch=%d loss=%.4f\r", m.Epoch, m.Loss)
	}
	fmt.Println()

	fmt.Println("===== FINAL PREDICTIONS =====")
	for i := 0; i < x.Rows; i++ {
		row := x.RowAt(i)
		input := nn.NewRowMat(row)

		logits := net.Infer(input)
		class := logits.ArgMax().Get(0, 0)
		fmt.Printf("%v | %v = %v\n", row[0], row[1], class)
	}
}
