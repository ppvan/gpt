package nn

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
