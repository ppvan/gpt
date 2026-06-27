package main

import (
	"fmt"

	"github.com/ppvan/gpt/nn"
)

func main() {

	data, err := nn.LoadCSV("cmd/ndigits/digits.csv", 64, false)
	if err != nil {
		panic(err)
	}

	data = data.Shuffle()

	train, val, _ := data.Split(0.80, 0.20, 0)

	model := nn.NewSequential(
		nn.NewLinear(64, 32),
		nn.LeakyRelu(0.01),
		nn.NewLinear(32, 64),
		nn.LeakyRelu(0.01),
		nn.NewLinear(64, 10),
		nn.LeakyRelu(0.01),
		nn.NewLinear(10, 10),
	)

	net := nn.NewNetwork(model, nn.CrossEntropy())

	epochs := 1000
	batchSize := 32

	for m := range net.Fit(train, epochs, batchSize) {
		fmt.Printf("epoch=%d loss=%.6f\r", m.Epoch, m.Loss)
	}

	fmt.Println()

	// ---- EVALUATION ----

	metrics := net.Evaluate(val)

	fmt.Println(metrics)
}
