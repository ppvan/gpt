package main

import (
	"fmt"
	"math/rand/v2"
)

var train = [][2]float32{
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
}

func cost(w float32) float32 {

	result := float32(0)
	for i := range train {
		x := train[i][0]
		y := train[i][1]
		y1 := x * w

		result += (y1 - y) * (y1 - y)
	}
	result /= float32(len(train))

	return result
}

func main() {
	// y = x*w
	w := 10 * rand.Float32()

	eps := float32(1e-3)
	rate := float32(0.001)

	fmt.Println("Cost", cost(w))

	for i := 0; i < 1000; i++ {
		dcost := (cost(w+eps) - cost(w)) / eps
		w = w - dcost*rate
		fmt.Println(w)
	}
	fmt.Println("Cost", cost(w))

}
