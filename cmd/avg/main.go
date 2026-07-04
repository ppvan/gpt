package main

import (
	"fmt"
	"math"
	"math/rand"

	nn "github.com/ppvan/gpt/pkg/mm"
)

// rnnParams holds all trainable weights/biases for the RNN.
type rnnParams struct {
	Wx, Wh, Wy nn.Mat
	bh, by     nn.Mat
}

// rnnCache holds the per-timestep values needed for backprop.
type rnnCache struct {
	xs []nn.Mat
	hs []nn.Mat // hs[0] = h_{-1}, hs[t+1] = h_t
	ys []nn.Mat
}

func makeRunningAverageDataset(seqLen int) (x nn.Mat, labels nn.Mat, maxVal float64) {
	maxVal = 100.0
	raw, targets := generateRunningAverageRaw(seqLen)
	scaledInput, scaledTarget := scaleSeries(raw, targets, maxVal)
	x = nn.NewRowMat(scaledInput).Transpose()
	labels = nn.NewRowMat(scaledTarget).Transpose()
	return
}

func generateRunningAverageRaw(seqLen int) (raw, targets []float64) {
	raw = make([]float64, seqLen)
	targets = make([]float64, seqLen)
	sum := 0.0
	for i := 0; i < seqLen; i++ {
		v := float64(rand.Intn(100))
		raw[i] = v
		sum += v
		targets[i] = sum / float64(i+1)
	}
	return
}

func scaleSeries(raw, targets []float64, maxVal float64) (scaledInput, scaledTarget []float64) {
	scaledInput = make([]float64, len(raw))
	scaledTarget = make([]float64, len(targets))
	for i := range raw {
		scaledInput[i] = raw[i] / maxVal
		scaledTarget[i] = targets[i] / maxVal
	}
	return
}

func newRNNParams(inputSize, hiddenSize, outputSize int) rnnParams {
	return rnnParams{
		Wx: nn.XavierMat(inputSize, hiddenSize),
		Wh: nn.XavierMat(hiddenSize, hiddenSize),
		Wy: nn.XavierMat(hiddenSize, outputSize),
		bh: nn.NewZeroMat(1, hiddenSize),
		by: nn.NewZeroMat(1, outputSize),
	}
}

// forwardPass runs the RNN over all timesteps and caches intermediate values.
func forwardPass(x nn.Mat, p rnnParams, timeSteps, hiddenSize int) rnnCache {
	h := nn.NewZeroMat(1, hiddenSize)
	cache := rnnCache{}
	cache.hs = append(cache.hs, h)
	for t := 0; t < timeSteps; t++ {
		xt := x.Row(t)
		h = xt.Multiply(p.Wx).Add(h.Multiply(p.Wh)).Add(p.bh).Apply(math.Tanh)
		y := h.Multiply(p.Wy).Add(p.by)
		cache.xs = append(cache.xs, xt)
		cache.hs = append(cache.hs, h)
		cache.ys = append(cache.ys, y)
	}
	return cache
}

// computeLoss sums 1/2 * (y_hat - y)^2 across all timesteps.
func computeLoss(cache rnnCache, labels nn.Mat) float64 {
	loss := 0.0
	for t := range cache.ys {
		target := labels.Row(t).Transpose()
		diff := cache.ys[t].Sub(target)
		loss += diff.Apply(func(f float64) float64 {
			return f * f * 0.5
		}).Sum()
	}
	return loss
}

// backwardPass runs BPTT and returns gradients for every parameter.
func backwardPass(cache rnnCache, labels nn.Mat, p rnnParams, timeSteps, inputSize, hiddenSize, outputSize int) rnnParams {
	grads := rnnParams{
		Wx: nn.NewZeroMat(inputSize, hiddenSize),
		Wh: nn.NewZeroMat(hiddenSize, hiddenSize),
		Wy: nn.NewZeroMat(hiddenSize, outputSize),
		bh: nn.NewZeroMat(1, hiddenSize),
		by: nn.NewZeroMat(1, outputSize),
	}
	dhNext := nn.NewZeroMat(1, hiddenSize)

	for t := timeSteps - 1; t >= 0; t-- {
		target := labels.Row(t)
		dyT := cache.ys[t].Sub(target)

		grads.by = grads.by.Add(dyT)
		grads.Wy = grads.Wy.Add(cache.hs[t+1].Transpose().Multiply(dyT))

		dhT := dyT.Multiply(p.Wy.Transpose()).Add(dhNext)
		tanhDeriv := cache.hs[t+1].Apply(func(h float64) float64 {
			return 1 - h*h
		})
		dzT := dhT.Hadamard(tanhDeriv)

		grads.Wx = grads.Wx.Add(cache.xs[t].Transpose().Multiply(dzT))
		grads.Wh = grads.Wh.Add(cache.hs[t].Transpose().Multiply(dzT))
		grads.bh = grads.bh.Add(dzT)

		dhNext = dzT.Multiply(p.Wh.Transpose())
	}
	return grads
}

// applyGradients updates params in-place using plain SGD.
func applyGradients(p *rnnParams, grads rnnParams, rate float64) {
	scale := func(f float64) float64 { return rate * f }
	p.Wx = p.Wx.Sub(grads.Wx.Apply(scale))
	p.Wh = p.Wh.Sub(grads.Wh.Apply(scale))
	p.bh = p.bh.Sub(grads.bh.Apply(scale))
	p.Wy = p.Wy.Sub(grads.Wy.Apply(scale))
	p.by = p.by.Sub(grads.by.Apply(scale))
}

// trainRNN runs the full training loop for the given number of epochs.
func trainRNN(x, labels nn.Mat, p *rnnParams, timeSteps, inputSize, hiddenSize, outputSize, epochs int, rate float64) {
	for range epochs {
		cache := forwardPass(x, *p, timeSteps, hiddenSize)
		loss := computeLoss(cache, labels)
		fmt.Println("loss:", loss)

		grads := backwardPass(cache, labels, *p, timeSteps, inputSize, hiddenSize, outputSize)
		applyGradients(p, grads, rate)
	}
}

// evaluate runs a forward pass and prints predicted vs actual running average.
func evaluate(x nn.Mat, p rnnParams, hiddenSize int, maxVal float64) {
	h := nn.NewZeroMat(1, hiddenSize)
	sum := 0.0
	for t := 0; t < x.Rows; t++ {
		xt := x.Row(t)
		h = xt.
			Multiply(p.Wx).
			Add(h.Multiply(p.Wh)).
			Add(p.bh).
			Apply(math.Tanh)
		sum += xt.Get(0, 0) * maxVal
		y := h.Multiply(p.Wy).Add(p.by)
		predicted := y.Get(0, 0) * maxVal
		actual := sum / float64(t+1)
		fmt.Printf(
			"step=%2d input=%6.1f predicted=%7.2f actual=%7.2f\n",
			t+1,
			xt.Get(0, 0)*maxVal,
			predicted,
			actual,
		)
	}
}

func main() {
	timeSteps := 20
	x, labels, maxVal := makeRunningAverageDataset(timeSteps)

	inputSize := 1
	outputSize := 1
	hiddenSize := 4
	epochs := 100000
	rate := 0.02

	params := newRNNParams(inputSize, hiddenSize, outputSize)

	trainRNN(x, labels, &params, timeSteps, inputSize, hiddenSize, outputSize, epochs, rate)

	evaluate(x, params, hiddenSize, maxVal)
}
