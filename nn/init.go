package nn

import "math/rand"

func weightInit() float64 {
	return rand.Float64()*2 - 1
}

func defaultOptimizerFactory() Optimizer {
	return &Gradient{Rate: 0.1}
}

func NewLayer(inSize, outSize int, act Activation, optFactory func() Optimizer) Layer {
	if optFactory == nil {
		optFactory = defaultOptimizerFactory
	}
	return Layer{
		Weights: NewZeroMat(inSize, outSize).Apply(func(f float64) float64 {
			return weightInit()
		}),
		Biases: NewZeroMat(1, outSize).Apply(func(f float64) float64 {
			return weightInit()
		}),
		Activation:      act,
		WeightOptimizer: optFactory(),
		BiasOptimizer:   optFactory(),
	}
}

func NewNetwork3(sizes []int) *Network {
	layers := make([]Layer, len(sizes)-1)
	for i := range layers {
		layers[i] = NewLayer(sizes[i], sizes[i+1], Sigmoid3{}, nil)
	}
	return &Network{Layers: layers, ErrorFunction: MSE{}}
}

func (nn *Network) WithErrorFunction(l ErrorFunction) *Network {
	nn.ErrorFunction = l
	return nn
}

func (nn *Network) WithOptimizer(optFactory func() Optimizer) *Network {
	for i := range nn.Layers {
		nn.Layers[i].WeightOptimizer = optFactory()
		nn.Layers[i].BiasOptimizer = optFactory()
	}
	return nn
}

func (nn *Network) WithActivation(act Activation) *Network {
	for i := range nn.Layers {
		nn.Layers[i].Activation = act
	}
	return nn
}
