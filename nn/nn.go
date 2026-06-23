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
