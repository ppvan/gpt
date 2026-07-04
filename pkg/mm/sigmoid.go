package mm

import "math"

type Sigmoid struct{}

type sigmoidCache struct {
	out Mat
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(x Mat) (out Mat, cache Cache) {
	out = x.Apply(func(v float64) float64 {
		return 1.0 / (1.0 + math.Exp(-v))
	})
	return out, sigmoidCache{out: out}
}

func (s *Sigmoid) Backward(cache Cache, dOut Mat) (dIn Mat, grads Grads) {
	c := cache.(sigmoidCache)

	dIn = dOut.Hadamard(
		c.out.Apply(func(y float64) float64 {
			return y * (1 - y)
		}),
	)

	return dIn, Grads{}
}
