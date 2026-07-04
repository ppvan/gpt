package main

import (
	"context"
	"fmt"

	"github.com/ppvan/gpt/pkg/mm"
)

func main() {
	data, err := mm.LoadCSV("cmd/ndigits/digits.csv", 64, false)
	if err != nil {
		panic(err)
	}
	data = data.Shuffle()
	train, val, _ := data.Split(0.80, 0.20, 0)

	model := mm.NewMultiLayerPerceptron(
		mm.NewLinear(64, 32),
		mm.NewLeakyReLU(0.02),
		mm.NewLinear(32, 64),
		mm.NewLeakyReLU(0.02),
		mm.NewLinear(64, 10),
		mm.NewLeakyReLU(0.02),
		mm.NewLinear(10, 10),
	)
	opt := mm.NewGradient(0.01)
	net := mm.NewNetwork(model, mm.CrossEntropy(), opt)

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
