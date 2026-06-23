package nn

import "math"

type Activation interface {
	Forward(x float64) float64
	Derivative(x float64) float64
}

type Sigmoid3 struct{}

func (s Sigmoid3) Forward(x float64) float64 {
	return float64(1 / (math.Exp(-x) + 1))
}
func (s Sigmoid3) Derivative(x float64) float64 {
	y := s.Forward(x)
	return y * (1 - y)
}

type sigmoid struct {
	lastOut Mat
}

func (s *sigmoid) Forward(x Mat) Mat {
	out := x.Apply(func(v float64) float64 {
		return 1.0 / (1.0 + math.Exp(-v))
	})
	s.lastOut = out
	return out
}

func (s *sigmoid) Backward(dOut Mat) Mat {
	// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
	return dOut.Hadamard(s.lastOut.Apply(func(v float64) float64 {
		return v * (1 - v)
	}))
}

func Sigmoid() *sigmoid {
	return &sigmoid{}
}
