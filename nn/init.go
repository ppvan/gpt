package nn

import "math/rand"

// weightInit returns a small random value in [-1, 1), used to
// initialize weights and biases.
func weightInit() float64 {
	return rand.Float64()*2 - 1
}

// NewLayer builds a fully-connected layer with randomly initialized
// weights and biases, ready to be used inside a Network.
//
// inSize is the number of inputs the layer accepts (i.e. the output
// size of the previous layer, or the feature count for the first layer).
// outSize is the number of neurons in this layer.
func NewLayer(inSize, outSize int, act Activation) Layer {

	return Layer{
		Weights: NewZeroMat(inSize, outSize).Apply(func(f float64) float64 {
			return weightInit()
		}),
		Biases: NewZeroMat(1, outSize).Apply(func(f float64) float64 {
			return weightInit()
		}),
		Activation: act,
	}
}

// NewNetwork builds a Network from a sequence of layers, defaulting
// to BinaryCrossEntrophy loss and Gradient descent optimization.
// Use WithLoss / WithOptimizer to override either.
//
// NewLayer should be used to construct each layer so that input/output
// sizes line up; mismatched layer dimensions will cause a runtime
// error during Train/Infer (matrix multiply shape mismatch).
func NewNetwork(layers ...Layer) *Network {
	return &Network{
		Layers:    layers,
		Loss:      BinaryCrossEntrophy{},
		Optimizer: Gradient{Rate: 0.5},
	}
}

// WithLoss sets the loss function and returns the network for chaining.
func (nn *Network) WithLoss(l Loss) *Network {
	nn.Loss = l
	return nn
}

// WithOptimizer sets the optimizer and returns the network for chaining.
func (nn *Network) WithOptimizer(o Optimizer) *Network {
	nn.Optimizer = o
	return nn
}
