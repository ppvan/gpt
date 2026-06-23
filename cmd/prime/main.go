package main

import (
	"fmt"

	"github.com/ppvan/gpt/nn"
)

func main() {
	primes()
}

func isPrime(n int) bool {
	if n < 2 {
		return false
	}
	for d := 2; d*d <= n; d++ {
		if n%d == 0 {
			return false
		}
	}
	return true
}

// features encodes an integer n as a small feature vector instead of
// just the raw number. A raw scalar (e.g. just "47") gives the
// network almost nothing to key off of, since primality isn't smooth
// w.r.t. n's value. Feeding n mod a handful of small primes mirrors
// what trial division actually checks, so the network has a much
// easier pattern to learn: "is the remainder against 2, 3, 5, 7 all
// nonzero?" is close to the real rule.
func features(n int) []float64 {
	mods := []int{2, 3, 5, 7, 11, 13}
	feat := make([]float64, 0, len(mods)+1)
	for _, m := range mods {
		// normalize remainder to [0,1) so all features are on a
		// similar scale, which helps gradient descent converge.
		feat = append(feat, float64(n%m)/float64(m))
	}
	// also include n itself, scaled down, as a weak extra signal
	feat = append(feat, float64(n)/100.0)
	return feat
}

func primes() {
	const (
		loInclusive = 2
		hiInclusive = 1000
	)

	var xRows [][]float64
	var yRows [][]float64
	for n := loInclusive; n <= hiInclusive; n++ {
		xRows = append(xRows, features(n))
		if isPrime(n) {
			yRows = append(yRows, []float64{1})
		} else {
			yRows = append(yRows, []float64{0})
		}
	}

	x := nn.NewMat(xRows)
	y := nn.NewMat(yRows)
	data := nn.NewDataset(x, y)

	// 7 input features -> two hidden layers -> 1 output (prime/not-prime)
	net := nn.NewNetwork3([]int{7, 16, 8, 1}).
		WithErrorFunction(nn.MSE{}). // swap for nn.BCE{} if you have a binary cross-entropy Loss impl
		WithOptimizer(func() nn.Optimizer { return &nn.Gradient{Rate: 0.5} })

	result := net.Train3(5000, data, nn.TrainConfig{
		BatchSize: 16,
		OnEpoch: func(epoch int, loss float64) {
			if epoch%500 == 0 {
				fmt.Printf("epoch %d: loss=%.6f\r", epoch, loss)
			}
		},
	})
	fmt.Println()
	fmt.Printf("final loss: %.6f\n\n", result.EpochLosses[len(result.EpochLosses)-1])

	fmt.Println("===== PREDICTIONS (2..99) =====")
	correct := 0
	total := 0
	for n := loInclusive; n <= hiInclusive; n++ {
		input := nn.NewRowMat(features(n))
		pred := net.Infer(input)
		predicted := pred.Get(0, 0)

		predictedPrime := predicted >= 0.5
		actualPrime := isPrime(n)

		marker := " "
		if predictedPrime != actualPrime {
			marker = "X" // flag mistakes for easy scanning
		} else {
			correct++
		}
		total++

		fmt.Printf("%s n=%2d  predicted=%.3f (%-5v)  actual=%-5v\n",
			marker, n, predicted, predictedPrime, actualPrime)
	}

	fmt.Printf("\nAccuracy: %d/%d (%.1f%%)\n", correct, total, 100*float64(correct)/float64(total))
}
