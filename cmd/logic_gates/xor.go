package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/ppvan/gpt/neural_network"
)

type Sigmoid struct{}

func (s Sigmoid) Forward(x float64) float64 {
	return float64(1 / (math.Exp(-x) + 1))
}
func (s Sigmoid) Derivative(x float64) float64 {
	y := s.Forward(x)
	return y * (1 - y)
}

type ReLU struct{}

func (r ReLU) Forward(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
func (r ReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

type BinaryCrossEntrophy struct{}

func (s BinaryCrossEntrophy) Loss(y, pred float64) float64 {
	return -y*math.Log(pred) - (1-y)*math.Log(1-pred)
}

func (s BinaryCrossEntrophy) Derivative(y, pred float64) float64 {
	pred = math.Max(pred, 1e-15)
	pred = math.Min(pred, 1-1e-15)

	return -y/pred + (1-y)/(1-pred)
}

type Gradient struct{}

func (g Gradient) Update(weight *float64, grad float64) {
	*weight -= grad * 0.5
}

var train = [][]float64{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}

// w * xW
func weight() float64 {
	return rand.Float64()*2 - 1
}

func main() {

	layer1 := neural_network.Layer{
		Weights: neural_network.Mat{
			Row:    2,
			Column: 2,
			Weights: [][]float64{
				{weight(), weight()},
				{weight(), weight()},
			},
		},
		Biases: neural_network.Mat{
			Row:    2,
			Column: 1,
			Weights: [][]float64{
				{weight()},
				{weight()},
			},
		},
		Activation: ReLU{},
	}
	layer2 := neural_network.Layer{
		Weights: neural_network.Mat{
			Row:    1,
			Column: 2,
			Weights: [][]float64{
				{weight(), weight()},
			},
		},
		Biases: neural_network.Mat{
			Row:    1,
			Column: 1,
			Weights: [][]float64{
				{weight()},
			},
		},
		Activation: Sigmoid{},
	}
	xor := neural_network.Network{
		Layers:    []neural_network.Layer{layer1, layer2},
		Loss:      BinaryCrossEntrophy{},
		Optimizer: Gradient{},
	}

	xor.Train(10000, train)

	for index := range train {
		row := train[index]
		x := row[:len(row)-1]

		pred := xor.Infer(x)

		fmt.Printf("%v | %v = %v\n", x[0], x[1], pred)
	}

}

// import (
// 	"fmt"
// 	"math"
// 	"math/rand"

// 	"github.com/ppvan/gpt/neural_network"
// )

// type Sigmoid struct{}

// func (s Sigmoid) Forward(x float64) float64 {
// 	return float64(1 / (math.Exp(-x) + 1))
// }

// func (s Sigmoid) Derivative(y float64) float64 {
// 	return y * (1 - y)
// }

// type BinaryCrossEntrophy struct{}

// func (s BinaryCrossEntrophy) Loss(y, pred float64) float64 {
// 	return -y*math.Log(pred) - (1-y)*math.Log(1-pred)
// }

// func (s BinaryCrossEntrophy) Derivative(y, pred float64) float64 {
// 	pred = math.Max(pred, 1e-15)
// 	pred = math.Min(pred, 1-1e-15)

// 	return -y/pred + (1-y)/(1-pred)
// }

// type Gradient struct{}

// func (g Gradient) Update(weight *float64, grad float64) {
// 	*weight -= grad * 0.5
// }

// var train = [][3]float64{
// 	{0, 0, 0},
// 	{0, 1, 1},
// 	{1, 0, 1},
// 	{1, 1, 0},
// }

// // w * x
// func Cost(nn neural_network.Network) float64 {

// 	result := float64(0)
// 	for i := range train {
// 		x1 := train[i][0]
// 		x2 := train[i][1]
// 		y := train[i][2]

// 		layer1 := nn.Layers[0]
// 		z1 := layer1.Weights[0][0]*x1 + layer1.Weights[0][1]*x2 + layer1.Biases[0]
// 		a1 := layer1.Activation.Forward(z1)

// 		z2 := layer1.Weights[1][0]*x1 + layer1.Weights[1][1]*x2 + layer1.Biases[1]
// 		a2 := layer1.Activation.Forward(z2)

// 		layer2 := nn.Layers[1]
// 		z3 := layer2.Weights[0][0]*a1 + layer2.Weights[0][1]*a2 + layer2.Biases[0]
// 		a3 := layer2.Activation.Forward(z3)

// 		result += nn.Loss.Loss(y, a3)
// 	}
// 	result /= float64(len(train))

// 	return result
// }

// func weight() float64 {
// 	return rand.Float64()*2 - 1
// }

// func main() {

// 	layer1 := neural_network.Layer{
// 		Weights:    [][]float64{{weight(), weight()}, {weight(), weight()}},
// 		Biases:     []float64{weight(), weight()},
// 		Activation: Sigmoid{},
// 	}
// 	layer2 := neural_network.Layer{
// 		Weights:    [][]float64{{weight(), weight()}},
// 		Biases:     []float64{weight()},
// 		Activation: Sigmoid{},
// 	}
// 	xor := neural_network.Network{
// 		Layers:    []neural_network.Layer{layer1, layer2},
// 		Loss:      BinaryCrossEntrophy{},
// 		Optimizer: Gradient{},
// 	}
// 	fmt.Println("Cost", Cost(xor))

// 	for range 200000 {
// 		c := float64(0)
// 		d_layer1_n1_w1 := float64(0)
// 		d_layer1_n1_w2 := float64(0)
// 		d_layer1_n1_b := float64(0)
// 		d_layer1_n2_w1 := float64(0)
// 		d_layer1_n2_w2 := float64(0)
// 		d_layer1_n2_b := float64(0)
// 		d_layer2_n1_w1 := float64(0)
// 		d_layer2_n1_w2 := float64(0)
// 		d_layer2_n1_b := float64(0)

// 		for i := range train {
// 			x1 := train[i][0]
// 			x2 := train[i][1]
// 			y := train[i][2]

// 			layer1 := xor.Layers[0]
// 			z1 := layer1.Weights[0][0]*x1 + layer1.Weights[0][1]*x2 + layer1.Biases[0]
// 			a1 := layer1.Activation.Forward(z1)

// 			z2 := layer1.Weights[1][0]*x1 + layer1.Weights[1][1]*x2 + layer1.Biases[1]
// 			a2 := layer1.Activation.Forward(z2)

// 			layer2 := xor.Layers[1]
// 			z3 := layer2.Weights[0][0]*a1 + layer2.Weights[0][1]*a2 + layer2.Biases[0]
// 			a3 := layer2.Activation.Forward(z3)

// 			l := xor.Loss.Loss(y, a3)
// 			c += l

// 			// l = loss(a3)
// 			// da3 = lost.Derivate(a3)
// 			// dz3 = layer2.Activation.Derivate(a3) * da3
// 			// d[layer2]w1 = a1 * dz3
// 			// d[layer2]w2 = a2 * dz3

// 			da3 := xor.Loss.Derivative(y, a3)
// 			dz3 := layer2.Activation.Derivative(a3) * da3
// 			d_layer2_n1_w1 += a1 * dz3
// 			d_layer2_n1_w2 += a2 * dz3
// 			d_layer2_n1_b += dz3

// 			// da2 = dz3 * d_z3_a2

// 			da2 := layer2.Weights[0][1] * dz3
// 			dz2 := layer1.Activation.Derivative(a2) * da2
// 			d_layer1_n2_w1 += x1 * dz2
// 			d_layer1_n2_w2 += x2 * dz2
// 			d_layer1_n2_b += dz2

// 			da1 := layer2.Weights[0][0] * dz3
// 			dz1 := layer1.Activation.Derivative(a1) * da1
// 			d_layer1_n1_w1 += x1 * dz1
// 			d_layer1_n1_w2 += x2 * dz1
// 			d_layer1_n1_b += dz1

// 		}
// 		c /= float64(len(train))
// 		d_layer1_n1_w1 /= float64(len(train))
// 		d_layer1_n1_w2 /= float64(len(train))
// 		d_layer1_n1_b /= float64(len(train))
// 		d_layer1_n2_w1 /= float64(len(train))
// 		d_layer1_n2_w2 /= float64(len(train))
// 		d_layer1_n2_b /= float64(len(train))
// 		d_layer2_n1_w1 /= float64(len(train))
// 		d_layer2_n1_w2 /= float64(len(train))
// 		d_layer2_n1_b /= float64(len(train))

// 		xor.Optimizer.Update(&xor.Layers[0].Weights[0][0], d_layer1_n1_w1)
// 		xor.Optimizer.Update(&xor.Layers[0].Weights[0][1], d_layer1_n1_w2)
// 		xor.Optimizer.Update(&xor.Layers[0].Biases[0], d_layer1_n1_b)

// 		xor.Optimizer.Update(&xor.Layers[0].Weights[1][0], d_layer1_n2_w1)
// 		xor.Optimizer.Update(&xor.Layers[0].Weights[1][1], d_layer1_n2_w2)
// 		xor.Optimizer.Update(&xor.Layers[0].Biases[1], d_layer1_n2_b)

// 		xor.Optimizer.Update(&xor.Layers[1].Weights[0][0], d_layer2_n1_w1)
// 		xor.Optimizer.Update(&xor.Layers[1].Weights[0][1], d_layer2_n1_w2)
// 		xor.Optimizer.Update(&xor.Layers[1].Biases[0], d_layer2_n1_b)

// 		fmt.Printf("Cost %v\r", c)
// 	}
// 	fmt.Println()

// 	for i := range train {
// 		x1 := train[i][0]
// 		x2 := train[i][1]
// 		layer1 := xor.Layers[0]
// 		z1 := layer1.Weights[0][0]*x1 + layer1.Weights[0][1]*x2 + layer1.Biases[0]
// 		a1 := layer1.Activation.Forward(z1)

// 		z2 := layer1.Weights[1][0]*x1 + layer1.Weights[1][1]*x2 + layer1.Biases[1]
// 		a2 := layer1.Activation.Forward(z2)

// 		layer2 := xor.Layers[1]
// 		z3 := layer2.Weights[0][0]*a1 + layer2.Weights[0][1]*a2 + layer2.Biases[0]
// 		a3 := layer2.Activation.Forward(z3)

// 		fmt.Printf("%v | %v = %v\n", x1, x2, a3)
// 	}

// }
