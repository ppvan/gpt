package main

import (
	"fmt"
	"math/rand"

	"github.com/ppvan/gpt/pkg/nn"
)

func makeRunningAverageDataset(seqLen int) (xs []nn.Mat, labels []nn.Mat, maxVal float64) {
	maxVal = 100.0
	raw := make([]float64, seqLen)
	targets := make([]float64, seqLen)
	sum := 0.0
	for i := 0; i < seqLen; i++ {
		v := float64(rand.Intn(100))
		raw[i] = v
		sum += v
		targets[i] = sum / float64(i+1)
	}

	xs = make([]nn.Mat, seqLen)
	labels = make([]nn.Mat, seqLen)
	for i := 0; i < seqLen; i++ {
		xs[i] = nn.NewMat(1, 1, []float64{raw[i] / maxVal})
		labels[i] = nn.NewMat(1, 1, []float64{targets[i] / maxVal})
	}
	return
}

func evaluate(model *nn.LSTM, xs []nn.Mat, hiddenSize int, maxVal float64) {
	h0 := nn.NewZeroMat(1, hiddenSize)
	c0 := nn.NewZeroMat(1, hiddenSize)
	outs, _, _, _ := model.Forward(xs, h0, c0)

	sum := 0.0
	for t, x := range xs {
		input := x.Get(0, 0) * maxVal
		sum += input
		predicted := outs[t].Get(0, 0) * maxVal
		actual := sum / float64(t+1)
		fmt.Printf(
			"step=%2d input=%6.1f predicted=%7.2f actual=%7.2f\n",
			t+1, input, predicted, actual,
		)
	}
}

func main() {
	timeSteps := 20
	inputSize := 1
	outputSize := 1
	hiddenSize := 4
	epochs := 100000
	rate := 0.02

	xs, labels, maxVal := makeRunningAverageDataset(timeSteps)

	model := nn.NewLSTM(inputSize, hiddenSize, outputSize)
	loss := nn.MeanSquareError()
	opt := nn.NewGradient(rate)

	h0 := nn.NewZeroMat(1, hiddenSize)
	c0 := nn.NewZeroMat(1, hiddenSize)

	for epoch := 0; epoch < epochs; epoch++ {
		outs, _, _, cache := model.Forward(xs, h0, c0)

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

		if epoch%1000 == 0 {
			fmt.Println("epoch:", epoch, "loss:", totalLoss)
		}
	}

	evaluate(model, xs, hiddenSize, maxVal)
}
