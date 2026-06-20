package nn

type Optimizer interface {
	Update(weight *float64, grad float64)
}

type Gradient struct {
	Rate float64
}

func (g Gradient) Update(weight *float64, grad float64) {
	*weight -= grad * g.Rate
}
