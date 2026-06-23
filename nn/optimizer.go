package nn

import "math"

type Optimizer interface {
	Update(weight *Mat, grad Mat)
}

type Gradient struct {
	Rate float64
}

func (g *Gradient) Update(weight *Mat, grad Mat) {
	*weight = weight.Sub(grad.Scale(g.Rate))
}

type Adam struct {
	Rate    float64
	Beta1   float64
	Beta2   float64
	Epsilon float64

	t    int // timestep, increments each Update call
	m, v Mat // first/second moment estimates
}

func NewAdam(rate float64) *Adam {
	return &Adam{
		Rate:    rate,
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
	}
}

func (a *Adam) Update(weight *Mat, grad Mat) {
	if a.t == 0 {
		a.m = NewZeroMat(grad.Rows, grad.Columns)
		a.v = NewZeroMat(grad.Rows, grad.Columns)
	}
	a.t++

	// m = beta1*m + (1-beta1)*grad
	a.m = a.m.Scale(a.Beta1).Add(grad.Scale(1 - a.Beta1))
	// v = beta2*v + (1-beta2)*grad^2
	gradSq := grad.Hadamard(grad)
	a.v = a.v.Scale(a.Beta2).Add(gradSq.Scale(1 - a.Beta2))

	// bias correction
	t := float64(a.t)
	mHat := a.m.Scale(1 / (1 - math.Pow(a.Beta1, t)))
	vHat := a.v.Scale(1 / (1 - math.Pow(a.Beta2, t)))

	// weight -= rate * mHat / (sqrt(vHat) + epsilon)
	denom := vHat.Apply(func(x float64) float64 { return math.Sqrt(x) }).Apply(
		func(x float64) float64 { return x + a.Epsilon },
	)
	step := mHat.Combine(denom, func(mv, dv float64) float64 { return mv / dv })
	*weight = weight.Sub(step.Scale(a.Rate))
}
