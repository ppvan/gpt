package nn

import (
	"math"
)

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

type relu struct {
	lastInput Mat
}

func (r *relu) Forward(x Mat) Mat {
	r.lastInput = x
	return x.Apply(func(v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	})
}

func (r *relu) Backward(dOut Mat) Mat {
	return dOut.Hadamard(
		r.lastInput.Apply(func(v float64) float64 {
			if v > 0 {
				return 1
			}
			return 0
		}),
	)
}

func ReLU() *relu {
	return &relu{}
}

type leakyRelu struct {
	lastInput Mat
	alpha     float64
}

func LeakyRelu(alpha float64) *leakyRelu {
	return &leakyRelu{
		alpha: alpha,
	}
}

func (r *leakyRelu) Forward(x Mat) Mat {
	r.lastInput = x

	return x.Apply(func(v float64) float64 {
		if v > 0 {
			return v
		}
		return r.alpha * v
	})
}

func (r *leakyRelu) Backward(dOut Mat) Mat {
	return dOut.Hadamard(
		r.lastInput.Apply(func(v float64) float64 {
			if v > 0 {
				return 1
			}
			return r.alpha
		}),
	)
}
