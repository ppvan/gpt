package main

import (
	"fmt"
	"math"
	"math/rand"
)

var train = [][3]float64{
	{0, 0, 0},
	{0, 1, 0},
	{1, 0, 0},
	{1, 1, 1},
}

func sigmoid(x float64) float64 {
	return float64(1 / (math.Exp(-x) + 1))
}

func cost(w1 float64, w2 float64, b float64) float64 {

	result := float64(0)
	for i := range train {
		x1 := train[i][0]
		x2 := train[i][1]
		y := train[i][2]
		y1 := sigmoid(w1*x1 + w2*x2 + b)

		result += (y1 - y) * (y1 - y)
	}
	result /= float64(len(train))

	return result
}

func main() {
	// y = x*w + b
	w1 := 10 * rand.Float64()
	w2 := 10 * rand.Float64()
	b := 10 * rand.Float64()

	eps := float64(0.1)
	rate := float64(0.1)

	fmt.Println("Cost", cost(w1, w2, b))

	for range 2_0_000 {
		c := cost(w1, w2, b)
		dw1 := (cost(w1+eps, w2, b) - c) / eps
		dw2 := (cost(w1, w2+eps, b) - c) / eps
		db := (cost(w1, w2, b+eps) - c) / eps
		w1 = w1 - dw1*rate
		w2 = w2 - dw2*rate
		b = b - db*rate
		// fmt.Println(cost(w1, w2, b))

	}
	fmt.Println(w1, w2, b)
	fmt.Println("Cost", cost(w1, w2, b))

	for i := range train {
		x1 := train[i][0]
		x2 := train[i][1]
		y1 := (w1*x1 + w2*x2 + b)

		fmt.Printf("%v | %v = %v\n", x1, x2, sigmoid(y1))
	}

}
