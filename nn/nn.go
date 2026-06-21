package nn

type Layer struct {
	Weights         Mat
	Biases          Mat
	Activation      Activation
	WeightOptimizer Optimizer
	BiasOptimizer   Optimizer
}

type Network struct {
	Layers []Layer

	ErrorFunction ErrorFunction
}

func (nn *Network) backprop(x, y Mat) (lost Mat, dW, db []Mat) {
	batchSize := x.Row
	last := len(nn.Layers) - 1

	z := make([]Mat, len(nn.Layers))
	a := make([]Mat, len(nn.Layers)+1)
	dW = make([]Mat, len(nn.Layers))
	db = make([]Mat, len(nn.Layers))
	delta := make([]Mat, len(nn.Layers))

	a[0] = x
	for l, layer := range nn.Layers {
		// z[l] = a[l] * w[l] + 1.b[l]
		one := NewZeroMat(batchSize, 1).Apply(func(f float64) float64 { return 1 })
		b := one.Multiply(layer.Biases)
		z[l] = a[l].Multiply(layer.Weights).Add(b)
		// a[l+1] = f[l](z[l]) - f(x) is the activation function at layer l
		a[l+1] = z[l].Apply(layer.Activation.Forward)
	}

	preds := a[last+1]
	lost = preds.Combine(y, func(predVal, yVal float64) float64 {
		return nn.ErrorFunction.Forward(yVal, predVal)
	})
	dLost := preds.Combine(y, func(predVal, yVal float64) float64 {
		return nn.ErrorFunction.Derivative(yVal, predVal)
	})

	delta[last] = dLost.Hadamard(
		z[last].Apply(nn.Layers[last].Activation.Derivative),
	)
	for j := last; j >= 0; j-- {
		// dW[j] = a[j]^T * delta[j]
		dW[j] = a[j].Transpose().Multiply(delta[j])
		// db[j] = 1^T * delta[j]
		oneT := NewZeroMat(1, batchSize).Apply(func(f float64) float64 { return 1 })
		db[j] = oneT.Multiply(delta[j])
		// delta[j-1] = delta[j] * w[j]^T ** f'[l](z[l-1])
		// f'(x) is the derivative of activation layer j
		if j > 0 {
			w := nn.Layers[j].Weights
			delta[j-1] = delta[j].Multiply(w.Transpose())
			delta[j-1] = delta[j-1].Hadamard(z[j-1].Apply(nn.Layers[j-1].Activation.Derivative))
		}
	}

	return lost, dW, db
}

func (nn *Network) learn(dW, db []Mat, n float64) {
	for j := range nn.Layers {
		nn.Layers[j].WeightOptimizer.Update(&nn.Layers[j].Weights, dW[j].Scale(1/n))
		nn.Layers[j].BiasOptimizer.Update(&nn.Layers[j].Biases, db[j].Scale(1/n))
	}
}

func (nn *Network) Infer(input Mat) Mat {
	a := input
	for _, layer := range nn.Layers {
		// a[i+1] = forward(w[i] * a[i] + b[i])
		one := NewZeroMat(a.Row, 1).Apply(func(f float64) float64 { return 1 })
		b := one.Multiply(layer.Biases)
		a = a.Multiply(layer.Weights).Add(b).Apply(layer.Activation.Forward)
	}
	return a
}

type TrainResult struct {
	EpochLosses []float64 // mean loss per epoch
}

type TrainConfig struct {
	BatchSize int

	OnEpoch func(epoch int, meanLoss float64)
}

// Train runs gradient descent over data for the given number of
// epochs, using mini-batches per TrainConfig. Returns per-epoch mean
// loss for later inspection/plotting.
func (nn *Network) Train(epochs int, data Dataset, cfg TrainConfig) TrainResult {
	loader := NewLoader(data, cfg.BatchSize)
	result := TrainResult{EpochLosses: make([]float64, 0, epochs)}

	for epoch := 0; epoch < epochs; epoch++ {
		batches := loader.NewEpoch()

		var epochLossSum float64
		var epochLossCount int

		for _, batch := range batches {
			lost, dW, db := nn.backprop(batch.X, batch.Y)
			nn.learn(dW, db, float64(batch.X.Row))

			epochLossSum += lost.Sum()
			epochLossCount += lost.Count()
		}

		meanLoss := epochLossSum / float64(epochLossCount)
		result.EpochLosses = append(result.EpochLosses, meanLoss)

		if cfg.OnEpoch != nil {
			cfg.OnEpoch(epoch, meanLoss)
		}
	}

	return result
}
