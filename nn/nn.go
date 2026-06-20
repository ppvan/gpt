package nn

type Layer struct {
	Weights Mat
	Biases  Mat

	Activation Activation
}

type Network struct {
	Layers []Layer

	Loss      Loss
	Optimizer Optimizer
}

func (nn *Network) outputSize() int {
	last := len(nn.Layers) - 1
	return nn.Layers[last].Weights.Row
}

// backprop runs forward + backward pass for a single training row and
// accumulates the weight/bias gradients into dW/db. It returns the
// per-output loss for that row so the caller can track epoch loss.
func (nn *Network) backprop(row []float64, dW, db []Mat) Mat {
	delta := make([]Mat, len(nn.Layers))
	raw_x := row[:len(row)-1]
	raw_y := row[len(row)-1:]
	last := len(nn.Layers) - 1
	x := NewRowMat(raw_x).Transpose()
	y := NewRowMat(raw_y).Transpose()

	a := make([]Mat, len(nn.Layers)+1)
	z := make([]Mat, len(nn.Layers)) // pre-activation values, one per layer
	a[0] = x
	for l, layer := range nn.Layers {
		z[l] = layer.Weights.Multiply(a[l]).Add(layer.Biases)
		a[l+1] = z[l].Apply(layer.Activation.Forward)
	}
	pred := a[len(a)-1]

	lost := pred.Apply(func(f float64) float64 {
		return nn.Loss.Loss(y.Weights[0][0], f)
	})

	dLoss := pred.Apply(func(f float64) float64 {
		return nn.Loss.Derivative(y.Weights[0][0], f)
	})

	// delta[last] = dLoss/dPred ⊙ activation'[last](z[last])
	delta[last] = dLoss.Hadamard(
		z[last].Apply(nn.Layers[last].Activation.Derivative),
	)

	for j := last; j >= 0; j-- {
		dW[j] = dW[j].Add(delta[j].Multiply(a[j].Transpose()))
		db[j] = db[j].Add(delta[j])
		if j > 0 {
			delta[j-1] = nn.Layers[j].Weights.Transpose().Multiply(delta[j])
			delta[j-1] = delta[j-1].Hadamard(
				z[j-1].Apply(nn.Layers[j-1].Activation.Derivative),
			)
		}
	}

	return lost
}

// learn applies accumulated gradients dW/db (summed over n samples) to the
// network's weights and biases.
func (nn *Network) learn(dW, db []Mat, n float64) {
	for j := range nn.Layers {
		for r := range nn.Layers[j].Weights.Weights {
			for col := range nn.Layers[j].Weights.Weights[r] {
				nn.Layers[j].Weights.Weights[r][col] -= 0.5 * dW[j].Weights[r][col] / n
			}
		}
		for r := range nn.Layers[j].Biases.Weights {
			for col := range nn.Layers[j].Biases.Weights[r] {
				nn.Layers[j].Biases.Weights[r][col] -= 0.5 * db[j].Weights[r][col] / n
			}
		}
	}
}

func (nn *Network) Train(epoch int, train_set Mat) {
	for range epoch {
		dW := make([]Mat, len(nn.Layers))
		db := make([]Mat, len(nn.Layers))
		for l := range nn.Layers {
			dW[l] = NewZeroMat(nn.Layers[l].Weights.Row, nn.Layers[l].Weights.Column)
			db[l] = NewZeroMat(nn.Layers[l].Biases.Row, nn.Layers[l].Biases.Column)
		}
		c := NewZeroMat(nn.outputSize(), 1)

		for index := range train_set.Row {
			lost := nn.backprop(train_set.Weights[index], dW, db)
			c = c.Add(lost)
		}

		n := float64(train_set.Row)
		c = c.Apply(func(f float64) float64 { return f / n })

		nn.learn(dW, db, n)

	}
}

// Infer runs a forward pass and returns the network's raw output values,
// one per output neuron. For single-output networks this is a length-1
// slice; callers that only ever had one output can do result[0].
func (nn *Network) Infer(input Mat) []float64 {

	a := input.Transpose()
	for _, layer := range nn.Layers {
		// a[i+1] = forward(w[i] * a[i] + b[i])
		a = layer.Weights.Multiply(a).Add(layer.Biases).Apply(layer.Activation.Forward)
	}

	out := make([]float64, len(a.Weights))
	for r := range a.Weights {
		out[r] = a.Weights[r][0]
	}

	return out
}
