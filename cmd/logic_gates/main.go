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
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	})
	y := nn.NewMat([][]float64{
		{0},
		{1},
		{1},
		{0},
	})

	xor := nn.NewNetwork([]int{2, 4, 4, 1})
	xor.Train(100000, x, y)

	fmt.Println("===== FINAL PREDICTIONS =====")
	for i := 0; i < x.Row; i++ {
		row := x.RowAt(i)
		input := nn.NewRowMat(row)
		pred := xor.Infer(input)
		fmt.Printf("%v | %v = %v\n", row[0], row[1], pred)
	}
}
