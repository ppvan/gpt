package mm

import "fmt"

type MultiLayerPerceptron struct {
	layers []Layer
}

func NewMultiLayerPerceptron(layers ...Layer) *MultiLayerPerceptron {
	return &MultiLayerPerceptron{layers: layers}
}

func (s *MultiLayerPerceptron) Add(l Layer) {
	s.layers = append(s.layers, l)
}

func (s *MultiLayerPerceptron) Forward(x Mat) (Mat, Cache) {
	caches := make([]Cache, len(s.layers))
	out := x
	for i, l := range s.layers {
		var c Cache
		out, c = l.Forward(out)
		caches[i] = c
	}
	return out, caches
}

func (s *MultiLayerPerceptron) Backward(cache Cache, dOut Mat) (Mat, Grads) {
	caches := cache.([]Cache)
	grads := Grads{}
	dIn := dOut

	for i := len(s.layers) - 1; i >= 0; i-- {
		var layerGrads Grads
		dIn, layerGrads = s.layers[i].Backward(caches[i], dIn)
		for name, g := range layerGrads {
			grads[layerKey(i, name)] = g
		}
	}
	return dIn, grads
}

func (s *MultiLayerPerceptron) Params() Params {
	params := Params{}
	for i, l := range s.layers {
		pl, ok := l.(ParamLayer)
		if !ok {
			continue
		}
		for name, p := range pl.Params() {
			params[layerKey(i, name)] = p
		}
	}
	return params
}

func (s *MultiLayerPerceptron) SetParams(p Params) {
	for i, l := range s.layers {
		pl, ok := l.(ParamLayer)
		if !ok {
			continue
		}
		layerParams := Params{}
		for name := range pl.Params() {
			layerParams[name] = p[layerKey(i, name)]
		}
		pl.SetParams(layerParams)
	}
}

func layerKey(index int, name string) string {
	return fmt.Sprintf("%d:%s", index, name)
}
