package nn

type LinearLayer struct {
	Weights Mat
	Biases  Mat

	weightOptimizer Optimizer
	biasOptimizer   Optimizer

	// cached during Forward, used in Backward
	lastInput Mat
}

func NewLinear(inputs, outputs int) *LinearLayer {
	return &LinearLayer{
		Weights:         HeMat(inputs, outputs),
		Biases:          HeMat(1, outputs),
		weightOptimizer: &Gradient{Rate: 0.01},
		biasOptimizer:   &Gradient{Rate: 0.01},
	}
}

func (l *LinearLayer) Forward(x Mat) Mat {
	l.lastInput = x
	one := NewZeroMat(x.Rows, 1).Apply(func(f float64) float64 { return 1 })
	b := one.Multiply(l.Biases)

	return x.Multiply(l.Weights).Add(b)
}

func (l *LinearLayer) Backward(dOut Mat) Mat {
	oneT := NewZeroMat(1, dOut.Rows).Apply(func(f float64) float64 { return 1 })
	dW := l.lastInput.Transpose().Multiply(dOut)
	dB := oneT.Multiply(dOut)

	dInput := dOut.Multiply(l.Weights.Transpose())

	// Update weights and biases via their optimizers
	l.Weights = l.weightOptimizer.Update(l.Weights, dW)
	l.Biases = l.biasOptimizer.Update(l.Biases, dB)

	return dInput
}
