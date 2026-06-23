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

func (n *Network) Train(epochs int, x, y Mat, callback func(epoch int, loss float64)) {
	for epoch := range epochs {

		out := n.model.Forward(x)
		loss := n.loss.Forward(out, y).Mean()
		dOut := n.loss.Backward(out, y)
		n.model.Backward(dOut)

		if callback != nil {
			callback(epoch, loss)
		}
	}
}
