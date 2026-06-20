package neural_network

type Activation interface {
	Forward(x float64) float64
	Derivative(y float64) float64
}

type Loss interface {
	Loss(y, pred float64) float64
	Derivative(y, pred float64) float64
}

type Optimizer interface {
	Update(weight *float64, grad float64)
}

type Layer struct {
	Weights [][]float64
	Biases  []float64

	Activation Activation
}

type Network struct {
	Layers []Layer

	Loss      Loss
	Optimizer Optimizer
}
