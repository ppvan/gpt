package neural_network

import (
	"fmt"
	"math"
)

type Activation interface {
	Forward(x float64) float64
	Derivative(x float64) float64
}

type Loss interface {
	Loss(y, pred float64) float64
	Derivative(y, pred float64) float64
}

type Optimizer interface {
	Update(weight *float64, grad float64)
}

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

type Sigmoid struct{}

func (s Sigmoid) Forward(x float64) float64 {
	return float64(1 / (math.Exp(-x) + 1))
}

func (s Sigmoid) Derivative(y float64) float64 {
	return y * (1 - y)
}

type BinaryCrossEntrophy struct{}

func (s BinaryCrossEntrophy) Loss(y, pred float64) float64 {
	return -y*math.Log(pred) - (1-y)*math.Log(1-pred)
}

func (s BinaryCrossEntrophy) Derivative(y, pred float64) float64 {
	pred = math.Max(pred, 1e-15)
	pred = math.Min(pred, 1-1e-15)

	return -y/pred + (1-y)/(1-pred)
}

type Gradient struct{}

func (g Gradient) Update(weight *float64, grad float64) {
	*weight -= grad * 0.5
}

func (nn Network) Cost(input []float64) float64 {
	return 0
}

func (nn *Network) Train(epoch int, train_set [][]float64) {
	for range epoch {
		dW := make([]Mat, len(nn.Layers))
		db := make([]Mat, len(nn.Layers))
		for l := range nn.Layers {
			dW[l] = NewZeroMat(nn.Layers[l].Weights.Row, nn.Layers[l].Weights.Column)
			db[l] = NewZeroMat(nn.Layers[l].Biases.Row, nn.Layers[l].Biases.Column)
		}
		c := NewRowMat([]float64{0})

		for index := range train_set {
			delta := make([]Mat, len(nn.Layers))
			row := train_set[index]
			x := row[:len(row)-1]
			y := row[len(row)-1:]
			last := len(nn.Layers) - 1
			x_mat := NewRowMat(x).Transpose()
			y_mat := NewRowMat(y).Transpose()

			a := make([]Mat, len(nn.Layers)+1)
			z := make([]Mat, len(nn.Layers)) // pre-activation values, one per layer
			a[0] = x_mat
			for l, layer := range nn.Layers {
				z[l] = layer.Weights.Multiply(a[l]).Add(layer.Biases)
				a[l+1] = z[l].Apply(layer.Activation.Forward)
			}
			pred := a[len(a)-1]

			lost := pred.Apply(func(f float64) float64 {
				return nn.Loss.Loss(y_mat.Weights[0][0], f)
			})
			c = c.Add(lost)

			// dLoss/dPred, elementwise — generic, works for any Loss
			dLoss := pred.Apply(func(f float64) float64 {
				return nn.Loss.Derivative(y_mat.Weights[0][0], f)
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

		n := float64(len(train_set))
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
		fmt.Println(c)
	}
}

func (nn *Network) Infer(input []float64) float64 {

	a := NewRowMat(input).Transpose()
	for _, layer := range nn.Layers {
		// a[i+1] = forward(w[i] * a[i] + b[i])
		a = layer.Weights.Multiply(a).Add(layer.Biases).Apply(layer.Activation.Forward)
	}

	pred := a.Weights[0][0]

	return pred
}
