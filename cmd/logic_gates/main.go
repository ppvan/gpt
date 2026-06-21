package main

import (
	"fmt"

	"github.com/ppvan/gpt/nn"
)

func main() {
	xor()

	// nand()
}

func nand() {
	train_set := nn.Mat{
		Row:    4,
		Column: 3,
		Weights: [][]float64{
			{0, 0, 1},
			{0, 1, 1},
			{1, 0, 1},
			{1, 1, 0},
		},
	}

	xor := nn.NewNetwork(
		nn.NewLayer(2, 4, nn.Sigmoid{}),
		nn.NewLayer(4, 4, nn.Sigmoid{}),
		nn.NewLayer(4, 1, nn.Sigmoid{}),
	)

	xor.OldTrain(10000, train_set)

	fmt.Println("===== FINAL PREDICTIONS =====")
	for index := range train_set.Weights {
		row := train_set.Weights[index]
		x := row[:len(row)-1]
		pred := xor.Infer(nn.NewRowMat(x))
		fmt.Printf("^(%v & %v) = %v\n", x[0], x[1], pred)
	}
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

	xor := nn.NewNetwork(
		nn.NewLayer(2, 4, nn.Sigmoid{}),
		nn.NewLayer(4, 4, nn.Sigmoid{}),
		nn.NewLayer(4, 1, nn.Sigmoid{}),
	)

	xor.Train(10000, x, y)

	fmt.Println("===== FINAL PREDICTIONS =====")
	for index := range x.Weights {
		row := x.Weights[index]
		x := nn.NewRowMat(row)
		pred := xor.Infer(x)
		fmt.Printf("%v | %v = %v\n", row[0], row[1], pred)
	}
}
