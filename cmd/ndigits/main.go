package main

import (
	"context"
	"fmt"

	"github.com/ppvan/gpt/pkg/nn"
)

func main() {
	data, err := nn.LoadCSV("cmd/ndigits/digits.csv", 64, false)
	if err != nil {
		panic(err)
	}
	data = data.Shuffle()
	train, val, _ := data.Split(0.80, 0.20, 0)

	model := nn.NewDense(
		nn.NewLinear(64, 32),
		nn.NewLeakyReLU(0.02),
		nn.NewLinear(32, 64),
		nn.NewLeakyReLU(0.02),
		nn.NewLinear(64, 10),
		nn.NewLeakyReLU(0.02),
		nn.NewLinear(10, 10),
	)
	opt := nn.NewGradient(0.01)
	net := nn.NewNetwork(model, nn.CrossEntropy(), opt)

	epochs := 2048
	batchSize := 32
	ctx := context.Background()
	for m := range net.Fit(ctx, train, epochs, batchSize) {
		fmt.Printf("epoch=%d loss=%.6f\r", m.Epoch, m.Loss)
	}
	fmt.Println()

	metrics, err := net.Evaluate(val)
	if err != nil {
		panic(err)
	}
	fmt.Println(metrics)
}
