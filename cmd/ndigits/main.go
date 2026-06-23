package main

import (
	"fmt"

	"github.com/ppvan/gpt/nn"
)

func main() {

	data, err := nn.LoadCSV("cmd/ndigits/digits.csv", 64, false)
	data = data.Shuffle().Transform(func(x, y nn.Mat) (nn.Mat, nn.Mat) {
		return x, y.OneHot(10)
	})

	if err != nil {
		panic(err)
	}

	train, _, _ := data.Split(0.75, 0.15, 0.10)

	model := nn.NewSequential(
		nn.NewLinear(64, 10),
		nn.Sigmoid(),
		nn.NewLinear(10, 10),
		nn.Sigmoid(),
		nn.NewLinear(10, 10),
	)

	net := nn.NewNetwork(model, nn.CrossEntropy())
	epochs := 10000
	batchSize := 10

	for m := range net.Fit(train, epochs, batchSize) {
		fmt.Printf("epoch=%d loss=%.4f\r", m.Epoch, m.Loss)
	}
	fmt.Println()
}
