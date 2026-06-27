package nn

type Module interface {
	Forward(x Mat) Mat
	Backward(dOut Mat) Mat
}

type Optimizer interface {
	Update(weight Mat, grad Mat) Mat
}

type LossFunction interface {
	Forward(pred, target Mat) Mat  // returns per-sample loss, shape (rows x 1)
	Backward(pred, target Mat) Mat // returns gradient w.r.t pred, shape same as pred
}
