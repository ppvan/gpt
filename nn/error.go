package nn

import "math"

type Loss interface {
	Loss(y, pred float64) float64
	Derivative(y, pred float64) float64
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

// MSE: Mean Squared Error loss (per-sample, not batch-averaged)
type MSE struct{}

func (m MSE) Loss(y, pred float64) float64 {
	diff := y - pred
	return diff * diff
}

func (m MSE) Derivative(y, pred float64) float64 {
	// d/dpred (y - pred)^2 = -2*(y - pred) = 2*(pred - y)
	return 2 * (pred - y)
}
