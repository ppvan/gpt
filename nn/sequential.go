package nn

type Sequential struct {
	Modules []Module
}

func NewSequential(modules ...Module) *Sequential {
	return &Sequential{Modules: modules}
}

func (s *Sequential) Forward(x Mat) Mat {
	out := x
	for _, m := range s.Modules {
		out = m.Forward(out)
	}
	return out
}

func (s *Sequential) Backward(dOut Mat) Mat {
	for i := len(s.Modules) - 1; i >= 0; i-- {
		dOut = s.Modules[i].Backward(dOut)
	}
	return dOut
}
