package nn

import "math"

type LossFunction interface {
	Forward(pred, target Mat) Mat  // returns per-sample loss, shape (rows x 1)
	Backward(pred, target Mat) Mat // returns gradient w.r.t pred, shape same as pred
}

type meanSquareError struct{}

func (m meanSquareError) Forward(pred, target Mat) Mat {
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

func (m meanSquareError) Backward(pred, target Mat) Mat {
	scale := 2.0 / float64(pred.Columns)
	out := NewZeroMat(pred.Rows, pred.Columns)
	for i := 0; i < pred.Rows; i++ {
		for j := 0; j < pred.Columns; j++ {
			out.Set(i, j, scale*(pred.Get(i, j)-target.Get(i, j)))
		}
	}
	return out
}

type crossEntropyLoss struct {
	probs Mat
}

func (c *crossEntropyLoss) Forward(logits, target Mat) Mat {
	out := NewZeroMat(logits.Rows, 1)
	c.probs = NewZeroMat(logits.Rows, logits.Columns) // cache here

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
			c.probs.Set(i, j, v) // store exp(x - max)
			sumExp += v
		}
		logSumExp := max + math.Log(sumExp)

		// normalize to get actual softmax probs
		for j := 0; j < logits.Columns; j++ {
			c.probs.Set(i, j, c.probs.Get(i, j)/sumExp)
		}

		t := int(target.Get(i, 0))
		loss := -(logits.Get(i, t) - logSumExp)
		out.Set(i, 0, loss)
	}
	return out
}

func (c *crossEntropyLoss) Backward(logits, target Mat) Mat {
	// reuse c.probs computed in Forward, no recomputation needed
	out := NewZeroMat(logits.Rows, logits.Columns)
	scale := 1.0 / float64(logits.Rows)
	for i := 0; i < logits.Rows; i++ {
		t := int(target.Get(i, 0))
		for j := 0; j < logits.Columns; j++ {
			grad := c.probs.Get(i, j)
			if j == t {
				grad -= 1.0
			}
			out.Set(i, j, scale*grad)
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

func MeanSquareError() *meanSquareError {

	return &meanSquareError{}
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
