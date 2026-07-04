package nn

type Linear struct {
	weights Mat
	biases  Mat
}

type linearCache struct {
	input Mat // x, needed for dW = x^T * dOut
}

func NewLinear(inputs, outputs int) *Linear {
	return &Linear{
		weights: HeMat(inputs, outputs),
		biases:  HeMat(1, outputs),
	}
}

func (l *Linear) Forward(x Mat) (out Mat, cache Cache) {
	one := NewZeroMat(x.Rows, 1).Apply(func(f float64) float64 { return 1 })
	b := one.Multiply(l.biases)
	out = x.Multiply(l.weights).Add(b)
	return out, linearCache{input: x}
}

func (l *Linear) Backward(cache Cache, dOut Mat) (dIn Mat, grads Grads) {
	c := cache.(linearCache)

	oneT := NewZeroMat(1, dOut.Rows).Apply(func(f float64) float64 { return 1 })

	dW := c.input.Transpose().Multiply(dOut)
	dB := oneT.Multiply(dOut)
	dIn = dOut.Multiply(l.weights.Transpose())

	grads = Grads{
		"W": dW,
		"b": dB,
	}
	return dIn, grads
}

func (l *Linear) Params() Params {
	return Params{
		"W": l.weights,
		"b": l.biases,
	}
}

func (l *Linear) SetParams(p Params) {
	l.weights = p["W"]
	l.biases = p["b"]
}
