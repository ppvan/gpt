package nn

type Optimizer interface {
	Update(weight Mat, grad Mat) Mat
}

type Gradient struct {
	Rate float64
}

func (g *Gradient) Update(weight Mat, grad Mat) Mat {
	w := weight.Sub(grad.Scale(g.Rate))

	return w
}
