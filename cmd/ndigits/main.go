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

	model := nn.NewSequential(
		nn.NewLinear(2, 4),
		nn.Sigmoid(),
		nn.NewLinear(4, 4),
		nn.Sigmoid(),
		nn.NewLinear(4, 2),
	)

	net := nn.NewNetwork(model, nn.CrossEntropy())

	net.Train(100000, x, y, func(epoch int, loss float64) {
		fmt.Printf("epoch %d: loss=%.6f\r", epoch, loss)
	})

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
