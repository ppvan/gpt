package nn

type EmbeddingLayer struct {
	Weights   Mat // [vocabSize × embeddingDim]
	optimizer Optimizer

	lastInput Mat // cached token IDs from Forward [batchSize × 1]
}

func NewEmbedding(vocabSize, embeddingDim int) *EmbeddingLayer {
	return &EmbeddingLayer{
		Weights:   randomMat(vocabSize, embeddingDim),
		optimizer: &Gradient{Rate: 0.01},
	}
}

func (e *EmbeddingLayer) Forward(x Mat) Mat {
	e.lastInput = x

	batchSize := x.Rows
	dim := e.Weights.Columns
	out := NewZeroMat(batchSize, dim)

	for i := range batchSize {
		tokenID := int(x.Get(i, 0))
		for j := range dim {
			out.Set(i, j, e.Weights.Get(tokenID, j))
		}
	}

	return out
}

func (e *EmbeddingLayer) Backward(dOut Mat) Mat {
	dW := NewZeroMat(e.Weights.Rows, e.Weights.Columns)

	batchSize := e.lastInput.Rows
	dim := e.Weights.Columns

	for i := range batchSize {
		tokenID := int(e.lastInput.Get(i, 0))
		for j := range dim {
			prev := dW.Get(tokenID, j)
			dW.Set(tokenID, j, prev+dOut.Get(i, j))
		}
	}

	e.Weights = e.optimizer.Update(e.Weights, dW)

	return dOut
}
