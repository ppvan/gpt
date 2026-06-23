package nn

type Network struct {
	model *Sequential
	loss  LossFunction
}

func NewNetwork(model *Sequential, loss LossFunction) *Network {
	return &Network{model: model, loss: loss}
}

func (n *Network) Infer(x Mat) Mat {
	return n.model.Forward(x)
}

func (n *Network) Predict(x Mat) Mat {
	logits := n.Infer(x)

	// output: batch x 1 (class index)
	out := NewZeroMat(logits.Rows, 1)

	for i := 0; i < logits.Rows; i++ {
		maxIdx := 0
		maxVal := logits.Get(i, 0)

		for j := 1; j < logits.Columns; j++ {
			v := logits.Get(i, j)
			if v > maxVal {
				maxVal = v
				maxIdx = j
			}
		}

		out.Set(i, 0, float64(maxIdx))
	}

	return out
}

func (n *Network) Fit(data Data, epochs int, batchSize int) <-chan EpochMetrics {
	out := make(chan EpochMetrics)

	go func() {
		defer close(out)

		for epoch := range epochs {

			epochLoss := 0.0
			batchCount := 0

			for _, batch := range data.Batches(batchSize) {

				x := batch.X
				y := batch.Y

				pred := n.model.Forward(x)
				loss := n.loss.Forward(pred, y).Mean()

				dOut := n.loss.Backward(pred, y)
				n.model.Backward(dOut)

				epochLoss += loss
				batchCount++
			}

			out <- EpochMetrics{
				Epoch: epoch + 1,
				Loss:  epochLoss / float64(batchCount),
			}
		}
	}()

	return out
}
