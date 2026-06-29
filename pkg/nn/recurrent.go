package nn

import "math"

type RNNLayer struct {
	// input-to-hidden weights:  shape (inputs, hidden)
	Wx Mat
	// hidden-to-hidden weights: shape (hidden, hidden)
	Wh Mat
	// hidden bias:              shape (1, hidden)
	Bh Mat

	wxOptimizer Optimizer
	whOptimizer Optimizer
	bhOptimizer Optimizer

	// cached during Forward, used in Backward
	lastInput  Mat
	lastHidden Mat
	lastPreAct Mat // pre-activation (before tanh): z = x·Wx + h·Wh + Bh
	Hidden     Mat // h_t, exposed so caller can pass it to next Forward
}

func NewRNN(inputs, hidden int) *RNNLayer {
	return &RNNLayer{
		Wx:          heMat(inputs, hidden),
		Wh:          heMat(hidden, hidden),
		Bh:          heMat(1, hidden),
		wxOptimizer: &Gradient{Rate: 0.01},
		whOptimizer: &Gradient{Rate: 0.01},
		bhOptimizer: &Gradient{Rate: 0.01},
		Hidden:      NewZeroMat(1, hidden), // h_0 = zeros
	}
}

// Forward computes one time step.
// x shape:      (1, inputs)
// returns h_t:  (1, hidden)
func (r *RNNLayer) Forward(x Mat) Mat {
	r.lastInput = x
	r.lastHidden = r.Hidden

	// broadcast bias across batch (same pattern as LinearLayer)
	one := NewZeroMat(x.Rows, 1).Apply(func(_ float64) float64 { return 1 })
	b := one.Multiply(r.Bh)

	// z = x·Wx + h_{t-1}·Wh + Bh
	z := x.Multiply(r.Wx).Add(r.lastHidden.Multiply(r.Wh)).Add(b)
	r.lastPreAct = z

	// h_t = tanh(z)
	r.Hidden = z.Apply(math.Tanh)
	return r.Hidden
}

// Backward receives dL/dh_t, returns dL/dx.
// It also accumulates dL/dh_{t-1} into r.Hidden gradient for BPTT —
// the caller is responsible for summing gradients across time steps.
func (r *RNNLayer) Backward(dHt Mat) Mat {
	// dL/dz = dL/dh_t ⊙ tanh'(z) = dL/dh_t ⊙ (1 - h_t²)
	hSq := r.Hidden.Apply(func(v float64) float64 { return v * v })
	dZ := dHt.Multiply(NewZeroMat(dHt.Rows, dHt.Columns).Apply(func(_ float64) float64 { return 1 }).Sub(hSq))

	one := NewZeroMat(1, dZ.Rows).Apply(func(_ float64) float64 { return 1 })

	// parameter gradients
	dWx := r.lastInput.Transpose().Multiply(dZ)  // (inputs, hidden)
	dWh := r.lastHidden.Transpose().Multiply(dZ) // (hidden, hidden)
	dBh := one.Multiply(dZ)                      // (1, hidden)

	// pass-through gradients
	dInput := dZ.Multiply(r.Wx.Transpose()) // (1, inputs)  → to previous layer
	dHPrev := dZ.Multiply(r.Wh.Transpose()) // (1, hidden)  → to previous time step

	// update parameters
	r.Wx = r.wxOptimizer.Update(r.Wx, dWx)
	r.Wh = r.whOptimizer.Update(r.Wh, dWh)
	r.Bh = r.bhOptimizer.Update(r.Bh, dBh)

	// store dHPrev so BPTT loop can retrieve it
	r.Hidden = dHPrev

	return dInput
}

// ResetHidden zeros the hidden state between sequences.
func (r *RNNLayer) ResetHidden() {
	r.Hidden = NewZeroMat(1, r.Wh.Rows)
}
