package nn

import "math"

type Activation interface {
	Forward(x float64) float64
	Derivative(x float64) float64
}

type Sigmoid struct{}

func (s Sigmoid) Forward(x float64) float64 {
	return float64(1 / (math.Exp(-x) + 1))
}
func (s Sigmoid) Derivative(x float64) float64 {
	y := s.Forward(x)
	return y * (1 - y)
}

type Tanh struct{}

func (t Tanh) Forward(x float64) float64 {
	return math.Tanh(x)
}

func (t Tanh) Derivative(x float64) float64 {
	// derivative of tanh(x) is 1 - tanh(x)^2
	th := math.Tanh(x)
	return 1 - th*th
}

type ReLU struct{}

func (r ReLU) Forward(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
func (r ReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
