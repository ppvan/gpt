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
	x := nn.Mat{
		Row:    4,
		Column: 2,
		Weights: [][]float64{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
		},
	}
	y := nn.Mat{
		Row:    4,
		Column: 1,
		Weights: [][]float64{
			{0},
			{1},
			{1},
			{0},
		},
	}

	xor := nn.NewNetwork([]int{2, 4, 4, 1})
	xor.Train(100000, x, y)

	fmt.Println("===== FINAL PREDICTIONS =====")
	for index := range x.Weights {
		row := x.Weights[index]
		input := nn.NewRowMat(row)
		pred := xor.Infer(input)
		fmt.Printf("%v | %v = %v\n", row[0], row[1], pred)
	}
}
