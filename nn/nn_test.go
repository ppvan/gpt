package nn

import "testing"

func BenchmarkTrainEpoch(b *testing.B) {
	data, err := LoadCSV("testdata/digits.csv", 64, false)
	if err != nil {
		b.Fatal(err)
	}

	train, val, test := data.Split(0.8, 0.2, 0)

	b.Logf(
		"train=%d val=%d test=%d",
		train.X.Rows,
		val.X.Rows,
		test.X.Rows,
	)

	model := NewSequential(
		NewLinear(64, 32),
		LeakyRelu(0.01),
		NewLinear(32, 64),
		LeakyRelu(0.01),
		NewLinear(64, 10),
		LeakyRelu(0.01),
		NewLinear(10, 10),
	)

	net := NewNetwork(model, CrossEntropy())

	for b.Loop() {
		for range net.Fit(train, 512, 32) {
		}
	}
}
