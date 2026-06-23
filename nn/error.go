package nn

import "math"

type ErrorFunction interface {
	Forward(y, pred float64) float64
	Derivative(y, pred float64) float64
}

type BCE struct{}

func (s BCE) Forward(y, pred float64) float64 {
	return -y*math.Log(pred) - (1-y)*math.Log(1-pred)
}

func (s BCE) Derivative(y, pred float64) float64 {
	pred = math.Max(pred, 1e-15)
	pred = math.Min(pred, 1-1e-15)

	return -y/pred + (1-y)/(1-pred)
}

type MSE struct{}

func (m MSE) Forward(y, pred float64) float64 {
	diff := y - pred
	return diff * diff
}

func (m MSE) Derivative(y, pred float64) float64 {
	// d/dpred (y - pred)^2 = -2*(y - pred) = 2*(pred - y)
	return 2 * (pred - y)
}

type LossFunction interface {
	Forward(pred, target Mat) Mat  // returns per-sample loss, shape (rows x 1)
	Backward(pred, target Mat) Mat // returns gradient w.r.t pred, shape same as pred
}

// MSE loss — for regression
type MSE2 struct{}

func (m MSE2) Forward(pred, target Mat) Mat {
	out := NewZeroMat(pred.Rows, 1)
	for i := 0; i < pred.Rows; i++ {
		sum := 0.0
		for j := 0; j < pred.Columns; j++ {
			d := pred.Get(i, j) - target.Get(i, j)
			sum += d * d
		}
		out.Set(i, 0, sum/float64(pred.Columns))
	}
	return out
}

func (m MSE2) Backward(pred, target Mat) Mat {
	scale := 2.0 / float64(pred.Columns)
	out := NewZeroMat(pred.Rows, pred.Columns)
	for i := 0; i < pred.Rows; i++ {
		for j := 0; j < pred.Columns; j++ {
			out.Set(i, j, scale*(pred.Get(i, j)-target.Get(i, j)))
		}
	}
	return out
}

type crossEntropyLoss struct{}

func (c crossEntropyLoss) Forward(logits, target Mat) Mat {
	out := NewZeroMat(logits.Rows, 1)
	for i := 0; i < logits.Rows; i++ {
		// log-sum-exp trick for stability
		max := logits.Get(i, 0)
		for j := 1; j < logits.Columns; j++ {
			if v := logits.Get(i, j); v > max {
				max = v
			}
		}
		sumExp := 0.0
		for j := 0; j < logits.Columns; j++ {
			sumExp += math.Exp(logits.Get(i, j) - max)
		}
		logSumExp := max + math.Log(sumExp)

		// NLL: -sum(target * log_softmax)
		loss := 0.0
		for j := 0; j < logits.Columns; j++ {
			if target.Get(i, j) > 0 { // skip zeros (one-hot)
				loss -= target.Get(i, j) * (logits.Get(i, j) - logSumExp)
			}
		}
		out.Set(i, 0, loss)
	}
	return out
}

func (c crossEntropyLoss) Backward(logits, target Mat) Mat {
	// Compute softmax(logits) first
	probs := NewZeroMat(logits.Rows, logits.Columns)
	for i := 0; i < logits.Rows; i++ {
		max := logits.Get(i, 0)
		for j := 1; j < logits.Columns; j++ {
			if v := logits.Get(i, j); v > max {
				max = v
			}
		}
		sumExp := 0.0
		for j := 0; j < logits.Columns; j++ {
			v := math.Exp(logits.Get(i, j) - max)
			probs.Set(i, j, v)
			sumExp += v
		}
		for j := 0; j < logits.Columns; j++ {
			probs.Set(i, j, probs.Get(i, j)/sumExp)
		}
	}

	// Gradient: (softmax(logits) - target) / batchSize
	out := NewZeroMat(logits.Rows, logits.Columns)
	scale := 1.0 / float64(logits.Rows)
	for i := 0; i < logits.Rows; i++ {
		for j := 0; j < logits.Columns; j++ {
			out.Set(i, j, scale*(probs.Get(i, j)-target.Get(i, j)))
		}
	}
	return out
}

func CrossEntropy() *crossEntropyLoss {

	return &crossEntropyLoss{}
}

func BinaryCrossEntropy() *binaryCrossEntropyLoss {

	return &binaryCrossEntropyLoss{}
}

type binaryCrossEntropyLoss struct{}

func (b binaryCrossEntropyLoss) Forward(pred, target Mat) Mat {
	out := NewZeroMat(pred.Rows, 1)

	for i := 0; i < pred.Rows; i++ {
		p := pred.Get(i, 0)

		// avoid log(0)
		p = math.Max(1e-15, math.Min(1-1e-15, p))

		y := target.Get(i, 0)

		loss :=
			-y*math.Log(p) -
				(1-y)*math.Log(1-p)

		out.Set(i, 0, loss)
	}

	return out
}

func (b binaryCrossEntropyLoss) Backward(pred, target Mat) Mat {
	out := NewZeroMat(pred.Rows, 1)

	scale := 1.0 / float64(pred.Rows)

	for i := 0; i < pred.Rows; i++ {
		p := pred.Get(i, 0)
		y := target.Get(i, 0)

		out.Set(i, 0,
			scale*(p-y)/(p*(1-p)))
	}

	return out
}
