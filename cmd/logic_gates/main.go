package main

import (
	"fmt"

	"github.com/ppvan/gpt/nn"
)

func main() {
	xor()

	nand()
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

	xor.Train(10000, train_set)

	fmt.Println("===== FINAL PREDICTIONS =====")
	for index := range train_set.Weights {
		row := train_set.Weights[index]
		x := row[:len(row)-1]
		pred := xor.Infer(nn.NewRowMat(x))
		fmt.Printf("^(%v & %v) = %v\n", x[0], x[1], pred)
	}
}

func xor() {
	train_set := nn.Mat{
		Row:    4,
		Column: 3,
		Weights: [][]float64{
			{0, 0, 0},
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

	xor.Train(10000, train_set)

	fmt.Println("===== FINAL PREDICTIONS =====")
	for index := range train_set.Weights {
		row := train_set.Weights[index]
		x := row[:len(row)-1]
		pred := xor.Infer(nn.NewRowMat(x))
		fmt.Printf("%v | %v = %v\n", x[0], x[1], pred)
	}
}
