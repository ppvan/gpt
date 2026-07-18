package nn

import "math"

type LSTMCell struct {
	Wf, Wi, Wo, Wg Mat
	Uf, Ui, Uo, Ug Mat
	Bf, Bi, Bo, Bg Mat
}

type lstmCellCache struct {
	x Mat

	hPrev Mat
	cPrev Mat

	f Mat
	i Mat
	o Mat
	g Mat

	c Mat
	h Mat
}

type lstmSeqCache struct {
	cellCaches []Cache
	outCaches  []Cache
}

type LSTM struct {
	Cell *LSTMCell

	Output *Linear

	hiddenSize int
}

func NewLSTMCell(inputSize, hiddenSize int) *LSTMCell {
	return &LSTMCell{
		Wf: XavierMat(inputSize, hiddenSize),
		Wi: XavierMat(inputSize, hiddenSize),
		Wo: XavierMat(inputSize, hiddenSize),
		Wg: XavierMat(inputSize, hiddenSize),

		Uf: XavierMat(hiddenSize, hiddenSize),
		Ui: XavierMat(hiddenSize, hiddenSize),
		Uo: XavierMat(hiddenSize, hiddenSize),
		Ug: XavierMat(hiddenSize, hiddenSize),

		Bf: NewZeroMat(1, hiddenSize),
		Bi: NewZeroMat(1, hiddenSize),
		Bo: NewZeroMat(1, hiddenSize),
		Bg: NewZeroMat(1, hiddenSize),
	}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (c *LSTMCell) Step(
	x Mat,
	hPrev Mat,
	cPrev Mat,
) (hNext Mat, cNext Mat, cache Cache) {

	zf := x.Dot(c.Wf).
		Add(hPrev.Dot(c.Uf)).
		AddBias(c.Bf)

	zi := x.Dot(c.Wi).
		Add(hPrev.Dot(c.Ui)).
		AddBias(c.Bi)

	zo := x.Dot(c.Wo).
		Add(hPrev.Dot(c.Uo)).
		AddBias(c.Bo)

	zg := x.Dot(c.Wg).
		Add(hPrev.Dot(c.Ug)).
		AddBias(c.Bg)

	f := zf.Apply(sigmoid)
	i := zi.Apply(sigmoid)
	o := zo.Apply(sigmoid)
	g := zg.Apply(math.Tanh)

	cNext = f.Hadamard(cPrev).
		Add(i.Hadamard(g))

	tanhC := cNext.Apply(math.Tanh)

	hNext = o.Hadamard(tanhC)

	cache = lstmCellCache{
		x: x,

		hPrev: hPrev,
		cPrev: cPrev,

		f: f,
		i: i,
		o: o,
		g: g,

		c: cNext,
		h: hNext,
	}

	return
}

func NewLSTM(
	inputSize,
	hiddenSize,
	outputSize int,
) *LSTM {

	return &LSTM{
		Cell: NewLSTMCell(
			inputSize,
			hiddenSize,
		),

		Output: NewLinear(
			hiddenSize,
			outputSize,
		),

		hiddenSize: hiddenSize,
	}
}

func (c *LSTMCell) BackwardStep(
	cache Cache,
	dh Mat,
	dcNext Mat,
) (
	dx Mat,
	dhPrev Mat,
	dcPrev Mat,
	grads Grads,
) {
	cc := cache.(lstmCellCache)

	tanhC := cc.c.Apply(math.Tanh)

	do := dh.Hadamard(tanhC)

	tanhDeriv := tanhC.Apply(func(v float64) float64 {
		return 1 - v*v
	})

	dc := dcNext.Add(
		dh.
			Hadamard(cc.o).
			Hadamard(tanhDeriv),
	)

	df := dc.Hadamard(cc.cPrev)

	di := dc.Hadamard(cc.g)

	dg := dc.Hadamard(cc.i)

	dcPrev = dc.Hadamard(cc.f)

	dzf := df.Hadamard(
		cc.f.Apply(func(v float64) float64 {
			return v * (1 - v)
		}),
	)

	dzi := di.Hadamard(
		cc.i.Apply(func(v float64) float64 {
			return v * (1 - v)
		}),
	)

	dzo := do.Hadamard(
		cc.o.Apply(func(v float64) float64 {
			return v * (1 - v)
		}),
	)

	dzg := dg.Hadamard(
		cc.g.Apply(func(v float64) float64 {
			return 1 - v*v
		}),
	)

	dWf := cc.x.Transpose().Dot(dzf)
	dWi := cc.x.Transpose().Dot(dzi)
	dWo := cc.x.Transpose().Dot(dzo)
	dWg := cc.x.Transpose().Dot(dzg)

	dUf := cc.hPrev.Transpose().Dot(dzf)
	dUi := cc.hPrev.Transpose().Dot(dzi)
	dUo := cc.hPrev.Transpose().Dot(dzo)
	dUg := cc.hPrev.Transpose().Dot(dzg)

	oneT := NewZeroMat(1, dzf.Rows).Apply(func(float64) float64 { return 1 })

	dBf := oneT.Dot(dzf)
	dBi := oneT.Dot(dzi)
	dBo := oneT.Dot(dzo)
	dBg := oneT.Dot(dzg)

	dx =
		dzf.Dot(c.Wf.Transpose()).
			Add(dzi.Dot(c.Wi.Transpose())).
			Add(dzo.Dot(c.Wo.Transpose())).
			Add(dzg.Dot(c.Wg.Transpose()))

	dhPrev =
		dzf.Dot(c.Uf.Transpose()).
			Add(dzi.Dot(c.Ui.Transpose())).
			Add(dzo.Dot(c.Uo.Transpose())).
			Add(dzg.Dot(c.Ug.Transpose()))

	grads = Grads{
		"Wf": dWf,
		"Wi": dWi,
		"Wo": dWo,
		"Wg": dWg,

		"Uf": dUf,
		"Ui": dUi,
		"Uo": dUo,
		"Ug": dUg,

		"bf": dBf,
		"bi": dBi,
		"bo": dBo,
		"bg": dBg,
	}

	return
}

func (l *LSTM) Forward(
	xs []Mat,
	h0 Mat,
	c0 Mat,
) (
	outs []Mat,
	hFinal Mat,
	cFinal Mat,
	cache Cache,
) {
	h := h0
	c := c0

	outs = make([]Mat, len(xs))

	sc := lstmSeqCache{}

	for t, x := range xs {

		hNext, cNext, cc :=
			l.Cell.Step(
				x,
				h,
				c,
			)

		y, oc := l.Output.Forward(hNext)

		outs[t] = y

		sc.cellCaches = append(sc.cellCaches, cc)
		sc.outCaches = append(sc.outCaches, oc)

		h = hNext
		c = cNext
	}

	return outs, h, c, sc
}

func (l *LSTM) Backward(
	cache Cache,
	dOuts []Mat,
) Grads {

	sc := cache.(lstmSeqCache)

	T := len(dOuts)

	dWf := NewZeroMat(l.Cell.Wf.Rows, l.Cell.Wf.Columns)
	dWi := NewZeroMat(l.Cell.Wi.Rows, l.Cell.Wi.Columns)
	dWo := NewZeroMat(l.Cell.Wo.Rows, l.Cell.Wo.Columns)
	dWg := NewZeroMat(l.Cell.Wg.Rows, l.Cell.Wg.Columns)

	dUf := NewZeroMat(l.Cell.Uf.Rows, l.Cell.Uf.Columns)
	dUi := NewZeroMat(l.Cell.Ui.Rows, l.Cell.Ui.Columns)
	dUo := NewZeroMat(l.Cell.Uo.Rows, l.Cell.Uo.Columns)
	dUg := NewZeroMat(l.Cell.Ug.Rows, l.Cell.Ug.Columns)

	dBf := NewZeroMat(l.Cell.Bf.Rows, l.Cell.Bf.Columns)
	dBi := NewZeroMat(l.Cell.Bi.Rows, l.Cell.Bi.Columns)
	dBo := NewZeroMat(l.Cell.Bo.Rows, l.Cell.Bo.Columns)
	dBg := NewZeroMat(l.Cell.Bg.Rows, l.Cell.Bg.Columns)

	dWy := NewZeroMat(
		l.Output.weights.Rows,
		l.Output.weights.Columns,
	)

	dBy := NewZeroMat(
		l.Output.biases.Rows,
		l.Output.biases.Columns,
	)

	dhNext := NewZeroMat(1, l.hiddenSize)
	dcNext := NewZeroMat(1, l.hiddenSize)

	for t := T - 1; t >= 0; t-- {

		dh, outGrads :=
			l.Output.Backward(
				sc.outCaches[t],
				dOuts[t],
			)

		dWy = dWy.Add(outGrads["W"])
		dBy = dBy.Add(outGrads["b"])

		dhTotal := dh.Add(dhNext)

		_, dhPrev, dcPrev, cellGrads :=
			l.Cell.BackwardStep(
				sc.cellCaches[t],
				dhTotal,
				dcNext,
			)

		dWf = dWf.Add(cellGrads["Wf"])
		dWi = dWi.Add(cellGrads["Wi"])
		dWo = dWo.Add(cellGrads["Wo"])
		dWg = dWg.Add(cellGrads["Wg"])

		dUf = dUf.Add(cellGrads["Uf"])
		dUi = dUi.Add(cellGrads["Ui"])
		dUo = dUo.Add(cellGrads["Uo"])
		dUg = dUg.Add(cellGrads["Ug"])

		dBf = dBf.Add(cellGrads["bf"])
		dBi = dBi.Add(cellGrads["bi"])
		dBo = dBo.Add(cellGrads["bo"])
		dBg = dBg.Add(cellGrads["bg"])

		dhNext = dhPrev
		dcNext = dcPrev
	}

	return Grads{
		"cell.Wf": dWf,
		"cell.Wi": dWi,
		"cell.Wo": dWo,
		"cell.Wg": dWg,

		"cell.Uf": dUf,
		"cell.Ui": dUi,
		"cell.Uo": dUo,
		"cell.Ug": dUg,

		"cell.bf": dBf,
		"cell.bi": dBi,
		"cell.bo": dBo,
		"cell.bg": dBg,

		"out.W": dWy,
		"out.b": dBy,
	}
}

func (c *LSTMCell) Params() Params {
	return Params{
		"Wf": c.Wf,
		"Wi": c.Wi,
		"Wo": c.Wo,
		"Wg": c.Wg,

		"Uf": c.Uf,
		"Ui": c.Ui,
		"Uo": c.Uo,
		"Ug": c.Ug,

		"bf": c.Bf,
		"bi": c.Bi,
		"bo": c.Bo,
		"bg": c.Bg,
	}
}

func (c *LSTMCell) SetParams(p Params) {
	c.Wf = p["Wf"]
	c.Wi = p["Wi"]
	c.Wo = p["Wo"]
	c.Wg = p["Wg"]

	c.Uf = p["Uf"]
	c.Ui = p["Ui"]
	c.Uo = p["Uo"]
	c.Ug = p["Ug"]

	c.Bf = p["bf"]
	c.Bi = p["bi"]
	c.Bo = p["bo"]
	c.Bg = p["bg"]
}

func (l *LSTM) Params() Params {
	p := Params{}

	for k, v := range l.Cell.Params() {
		p["cell."+k] = v
	}

	for k, v := range l.Output.Params() {
		p["out."+k] = v
	}

	return p
}

func (l *LSTM) SetParams(p Params) {
	l.Cell.SetParams(Params{
		"Wf": p["cell.Wf"],
		"Wi": p["cell.Wi"],
		"Wo": p["cell.Wo"],
		"Wg": p["cell.Wg"],

		"Uf": p["cell.Uf"],
		"Ui": p["cell.Ui"],
		"Uo": p["cell.Uo"],
		"Ug": p["cell.Ug"],

		"bf": p["cell.bf"],
		"bi": p["cell.bi"],
		"bo": p["cell.bo"],
		"bg": p["cell.bg"],
	})

	l.Output.SetParams(Params{
		"W": p["out.W"],
		"b": p["out.b"],
	})
}
