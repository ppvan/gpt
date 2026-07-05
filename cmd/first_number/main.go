package main

import (
	"fmt"
	"math/rand"

	"github.com/ppvan/gpt/pkg/nn"
)

func makeRememberFirstDataset(seqLen int) (xs []nn.Mat, labels []nn.Mat, maxVal float64) {
	maxVal = 100.0

	first := float64(rand.Intn(100))

	xs = make([]nn.Mat, seqLen)
	labels = make([]nn.Mat, seqLen)

	// first input
	xs[0] = nn.NewMat(1, 1, []float64{first / maxVal})
	labels[0] = nn.NewZeroMat(1, 1)

	// remaining inputs are random noise
	for i := 1; i < seqLen; i++ {
		v := float64(rand.Intn(100))
		xs[i] = nn.NewMat(1, 1, []float64{v / maxVal})
		labels[i] = nn.NewZeroMat(1, 1)
	}

	labels[seqLen-1] = nn.NewMat(1, 1, []float64{first / maxVal})

	return
}

func evaluate(model *nn.Recurent, xs []nn.Mat, hiddenSize int, maxVal float64) {
	h0 := nn.NewZeroMat(1, hiddenSize)
	outs, _, _ := model.Forward(xs, h0)

	fmt.Println("Sequence")

	for t := range xs {
		input := xs[t].Get(0, 0) * maxVal
		target := 0.0
		if t == len(xs)-1 {
			target = xs[0].Get(0, 0) * maxVal
		}

		pred := outs[t].Get(0, 0) * maxVal

		fmt.Printf(
			"step=%2d input=%6.1f pred=%7.2f target=%7.2f\n",
			t+1,
			input,
			pred,
			target,
		)
	}
}

func main() {
	timeSteps := 20 // Try 10, 20, 50, 100, 200
	inputSize := 1
	outputSize := 1
	hiddenSize := 8

	epochs := 2000
	learningRate := 0.002

	xs, labels, maxVal := makeRememberFirstDataset(timeSteps)

	model := nn.NewRecurent(inputSize, hiddenSize, outputSize)

	loss := nn.MeanSquareError()
	opt := nn.NewGradient(learningRate)

	h0 := nn.NewZeroMat(1, hiddenSize)

	for epoch := 0; epoch < epochs; epoch++ {

		outs, _, cache := model.Forward(xs, h0)

		totalLoss := 0.0
		dOuts := make([]nn.Mat, timeSteps)

		for t := 0; t < timeSteps; t++ {

			l, err := loss.Forward(outs[t], labels[t])
			if err != nil {
				panic(err)
			}

			totalLoss += l.Get(0, 0)

			dOut, err := loss.Backward(outs[t], labels[t])
			if err != nil {
				panic(err)
			}

			dOuts[t] = dOut
		}

		grads := model.Backward(cache, dOuts)

		params := model.Params()
		newParams := nn.Params{}

		for name, p := range params {
			newParams[name] = opt.Update(name, p, grads[name])
		}

		model.SetParams(newParams)

		if epoch%100 == 0 {
			fmt.Printf("epoch=%6d loss=%f\n", epoch, totalLoss)
		}
	}

	evaluate(model, xs, hiddenSize, maxVal)
}
