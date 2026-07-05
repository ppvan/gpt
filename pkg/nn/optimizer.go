package nn

import "math"

type Gradient struct {
	Rate float64
}

func NewGradient(rate float64) *Gradient {
	return &Gradient{Rate: rate}
}

func (g *Gradient) Update(name string, param Mat, grad Mat) Mat {
	return param.Sub(grad.Scale(g.Rate))
}

type Momentum struct {
	gradAvg map[string]Mat

	Rate float64
	Beta float64
}

func NewMomentum(rate float64, beta float64) *Momentum {
	return &Momentum{Rate: rate, Beta: beta, gradAvg: make(map[string]Mat)}
}

func (g *Momentum) Update(name string, param Mat, grad Mat) Mat {
	v := g.gradAvg[name]
	if v.Rows == 0 {
		v = NewZeroMat(grad.Rows, grad.Columns)
	}

	v = v.Scale(g.Beta).
		Add(grad.Scale(1 - g.Beta))

	g.gradAvg[name] = v

	return param.Sub(v.Scale(g.Rate))
}

type RMSProp struct {
	avgSq map[string]Mat

	Rate    float64
	Beta    float64
	Epsilon float64
}

func NewRMSProp(rate, beta, eps float64) *RMSProp {
	return &RMSProp{
		avgSq:   make(map[string]Mat),
		Rate:    rate,
		Beta:    beta,
		Epsilon: eps,
	}
}

func (r *RMSProp) Update(name string, param, grad Mat) Mat {
	s, ok := r.avgSq[name]
	if !ok {
		s = NewZeroMat(grad.Rows, grad.Columns)
	}

	// grad²
	grad2 := grad.Hadamard(grad)

	// running average
	s = s.Scale(r.Beta).
		Add(grad2.Scale(1 - r.Beta))

	r.avgSq[name] = s

	// sqrt(s) + eps
	term := s.
		Apply(math.Sqrt).
		Apply(func(v float64) float64 {
			return 1 / (v + r.Epsilon)
		})

	update := grad.Hadamard(term)

	return param.Sub(update.Scale(r.Rate))
}
