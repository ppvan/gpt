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
	y := nn.NewMat([][]float64{
		{0}, {1}, {1}, {0},
	})

	data := nn.NewDataset(x, y)
	xor := nn.NewNetwork([]int{2, 4, 4, 1})

	result := xor.Train(100000, data, nn.TrainConfig{
		BatchSize: 0, // full-batch, same behavior as before
		OnEpoch: func(epoch int, loss float64) {
			if epoch%10000 == 0 {
				fmt.Printf("epoch %d: loss=%.6f\r", epoch, loss)
			}
		},
	})
	fmt.Println()

	fmt.Println("final loss:", result.EpochLosses[len(result.EpochLosses)-1])

	fmt.Println("===== FINAL PREDICTIONS =====")
	for i := 0; i < x.Rows; i++ {
		row := x.RowAt(i)
		input := nn.NewRowMat(row)
		pred := xor.Infer(input)
		fmt.Printf("%v | %v = %v\n", row[0], row[1], pred)
	}
}
