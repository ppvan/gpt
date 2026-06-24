package nn

type EpochMetrics struct {
	Epoch int
	Loss  float64
}

func Accuracy(pred, y Mat) float64 {
	correct := 0
	n := pred.Rows

	for i := range n {
		p := int(pred.Get(i, 0))
		t := int(y.Get(i, 0))

		if p == t {
			correct++
		}
	}

	return float64(correct) / float64(n)
}
