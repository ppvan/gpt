package main

import (
	"context"
	"fmt"

	"github.com/ppvan/gpt/pkg/nn"
)

func main() {
	xor()
	// nand()
}

func xor() {
	x := nn.NewMat(
		4, 2, []float64{0, 0, 0, 1, 1, 0, 1, 1},
	)
	labels := nn.NewMat(4, 1, []float64{0, 1, 1, 0})
	data := nn.Data{
		X: x, Y: labels,
	}
	model := nn.NewDense(
		nn.NewLinear(2, 4),
		nn.NewLeakyReLU(0.1),
		nn.NewLinear(4, 16),
		nn.NewLeakyReLU(0.1),
		nn.NewLinear(16, 2),
	)
	opt := nn.NewGradient(0.01)
	net := nn.NewNetwork(model, nn.CrossEntropy(), opt)

	ctx := context.Background()
	for m := range net.Fit(ctx, data, 5000, 4) {
		fmt.Printf("epoch=%d loss=%.4f\r", m.Epoch, m.Loss)
	}
	fmt.Println()

	fmt.Println("===== FINAL PREDICTIONS =====")
	for i := 0; i < x.Rows; i++ {
		input := x.Slice(i, i+1)
		pred, err := net.Predict(input)
		if err != nil {
			fmt.Printf("%v | %v = error: %v\n", input.Get(0, 0), input.Get(0, 1), err)
			continue
		}
		fmt.Printf("%v | %v = %d\n", input.Get(0, 0), input.Get(0, 1), pred.Class)
	}
}
