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

func (nn *Network) Train(epoch int, train_set Mat) {
	for range epoch {
		dW := make([]Mat, len(nn.Layers))
		db := make([]Mat, len(nn.Layers))
		for l := range nn.Layers {
			dW[l] = NewZeroMat(nn.Layers[l].Weights.Row, nn.Layers[l].Weights.Column)
			db[l] = NewZeroMat(nn.Layers[l].Biases.Row, nn.Layers[l].Biases.Column)
		}
		c := NewRowMat([]float64{0})

		for index := range train_set.Row {
			delta := make([]Mat, len(nn.Layers))
			row := train_set.Weights[index]
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
			c = c.Add(lost)

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
		}

		n := float64(train_set.Row)
		c = c.Apply(func(f float64) float64 { return f / n })

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
}

func (nn *Network) Infer(input Mat) float64 {

	a := input.Transpose()
	for _, layer := range nn.Layers {
		// a[i+1] = forward(w[i] * a[i] + b[i])
		a = layer.Weights.Multiply(a).Add(layer.Biases).Apply(layer.Activation.Forward)
	}

	pred := a.Weights[0][0]

	return pred
}
