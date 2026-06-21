package nn

import "math"

type ErrorFunction interface {
	Forward(y, pred float64) float64
	Derivative(y, pred float64) float64
}

type BCE struct{}

func (s BCE) Forward(y, pred float64) float64 {
	return -y*math.Log(pred) - (1-y)*math.Log(1-pred)
}

func (s BCE) Derivative(y, pred float64) float64 {
	pred = math.Max(pred, 1e-15)
	pred = math.Min(pred, 1-1e-15)

	return -y/pred + (1-y)/(1-pred)
}

type MSE struct{}

func (m MSE) Forward(y, pred float64) float64 {
	diff := y - pred
	return diff * diff
}

func (m MSE) Derivative(y, pred float64) float64 {
	// d/dpred (y - pred)^2 = -2*(y - pred) = 2*(pred - y)
	return 2 * (pred - y)
}
