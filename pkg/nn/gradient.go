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
