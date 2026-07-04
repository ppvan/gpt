package nn

import "math"

type RecurentCell struct {
	Wx, Wh, Bh Mat
}

type rnnSeqCache struct {
	cellCaches []Cache
	outCaches  []Cache
}

type rnnCellCache struct {
	x, hPrev, hNext Mat
}

type Recurent struct {
	Cell       *RecurentCell
	Output     *Linear
	hiddenSize int
}

func NewRecurentCell(inputSize, hiddenSize int) *RecurentCell {
	return &RecurentCell{
		Wx: XavierMat(inputSize, hiddenSize),
		Wh: XavierMat(hiddenSize, hiddenSize),
		Bh: NewZeroMat(1, hiddenSize),
	}
}

func (c *RecurentCell) Step(x, hPrev Mat) (hNext Mat, cache Cache) {
	z := x.Dot(c.Wx).Add(hPrev.Dot(c.Wh)).Add(c.Bh)
	hNext = z.Apply(math.Tanh)
	return hNext, rnnCellCache{x: x, hPrev: hPrev, hNext: hNext}
}

func (c *RecurentCell) BackwardStep(cache Cache, dhNext Mat) (dx, dhPrev Mat, grads Grads) {
	cc := cache.(rnnCellCache)

	tanhDeriv := cc.hNext.Apply(func(h float64) float64 { return 1 - h*h })
	dz := dhNext.Hadamard(tanhDeriv)

	dWx := cc.x.Transpose().Dot(dz)
	dWh := cc.hPrev.Transpose().Dot(dz)
	dBh := dz

	dx = dz.Dot(c.Wx.Transpose())
	dhPrev = dz.Dot(c.Wh.Transpose())

	grads = Grads{"Wx": dWx, "Wh": dWh, "bh": dBh}
	return dx, dhPrev, grads
}

func (c *RecurentCell) Params() Params {
	return Params{"Wx": c.Wx, "Wh": c.Wh, "bh": c.Bh}
}

func (c *RecurentCell) SetParams(p Params) {
	c.Wx = p["Wx"]
	c.Wh = p["Wh"]
	c.Bh = p["bh"]
}

func NewRecurent(inputSize, hiddenSize, outputSize int) *Recurent {
	return &Recurent{
		Cell:       NewRecurentCell(inputSize, hiddenSize),
		Output:     NewLinear(hiddenSize, outputSize),
		hiddenSize: hiddenSize,
	}
}

func (r *Recurent) Forward(xs []Mat, h0 Mat) (outs []Mat, hFinal Mat, cache Cache) {
	h := h0
	outs = make([]Mat, len(xs))
	sc := rnnSeqCache{}

	for t, x := range xs {
		hNext, cc := r.Cell.Step(x, h)
		y, oc := r.Output.Forward(hNext)

		outs[t] = y
		sc.cellCaches = append(sc.cellCaches, cc)
		sc.outCaches = append(sc.outCaches, oc)
		h = hNext
	}
	return outs, h, sc
}

func (r *Recurent) Backward(cache Cache, dOuts []Mat) Grads {
	sc := cache.(rnnSeqCache)
	T := len(dOuts)

	dWx := NewZeroMat(r.Cell.Wx.Rows, r.Cell.Wx.Columns)
	dWh := NewZeroMat(r.Cell.Wh.Rows, r.Cell.Wh.Columns)
	dBh := NewZeroMat(r.Cell.Bh.Rows, r.Cell.Bh.Columns)
	dWy := NewZeroMat(r.Output.weights.Rows, r.Output.weights.Columns)
	dBy := NewZeroMat(r.Output.biases.Rows, r.Output.biases.Columns)

	dhNext := NewZeroMat(1, r.hiddenSize)

	for t := T - 1; t >= 0; t-- {
		dh, outGrads := r.Output.Backward(sc.outCaches[t], dOuts[t])
		dWy = dWy.Add(outGrads["W"])
		dBy = dBy.Add(outGrads["b"])

		dhTotal := dh.Add(dhNext)
		_, dhPrev, cellGrads := r.Cell.BackwardStep(sc.cellCaches[t], dhTotal)

		dWx = dWx.Add(cellGrads["Wx"])
		dWh = dWh.Add(cellGrads["Wh"])
		dBh = dBh.Add(cellGrads["bh"])
		dhNext = dhPrev
	}

	return Grads{
		"cell.Wx": dWx, "cell.Wh": dWh, "cell.bh": dBh,
		"out.W": dWy, "out.b": dBy,
	}
}

func (r *Recurent) Params() Params {
	p := Params{}
	for k, v := range r.Cell.Params() {
		p["cell."+k] = v
	}
	for k, v := range r.Output.Params() {
		p["out."+k] = v
	}
	return p
}

func (r *Recurent) SetParams(p Params) {
	r.Cell.SetParams(Params{"Wx": p["cell.Wx"], "Wh": p["cell.Wh"], "bh": p["cell.bh"]})
	r.Output.SetParams(Params{"W": p["out.W"], "b": p["out.b"]})
}
