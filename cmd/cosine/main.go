package main

import (
	"fmt"
	"image/color"
	"math"

	"github.com/ppvan/gpt/nn"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {

	const samples = 100

	trainSet := nn.Mat{
		Row:     samples,
		Column:  2,
		Weights: make([][]float64, samples),
	}

	for i := 0; i < samples; i++ {
		x := 2 * math.Pi * float64(i) / float64(samples-1)

		trainSet.Weights[i] = []float64{
			x,
			math.Cos(x),
		}
	}

	xor := nn.NewNetwork(
		nn.NewLayer(1, 4, nn.Sigmoid{}),
		nn.NewLayer(4, 4, nn.Sigmoid{}),
		nn.NewLayer(4, 1, nn.Tanh{}),
	).WithOptimizer(nn.Gradient{Rate: 0.1}).WithLoss(nn.MSE{})

	xor.OldTrain(10000, trainSet)

	fmt.Println("===== FINAL PREDICTIONS =====")
	for index := range trainSet.Weights {
		row := trainSet.Weights[index]
		x := row[:len(row)-1]
		pred := xor.Infer(nn.NewRowMat(x))
		fmt.Printf("cos(%v) = %v = |%v\n", x[0], pred, math.Cos(x[0]))
	}

	p := plot.New()
	p.Title.Text = "cos(x) vs Neural Net Prediction"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	actual := make(plotter.XYs, 200)
	predicted := make(plotter.XYs, 200)

	for i := 0; i < 200; i++ {
		x := -2*math.Pi + float64(i)*(4*math.Pi/200)
		actual[i].X = x
		actual[i].Y = math.Cos(x)

		predicted[i].X = x
		predicted[i].Y = xor.Infer(nn.NewRowMat([]float64{x}))[0]
	}

	actualLine, _ := plotter.NewLine(actual)
	actualLine.Color = color.Black

	predLine, _ := plotter.NewLine(predicted)
	predLine.Dashes = []vg.Length{vg.Points(4), vg.Points(4)} // dashed to distinguish

	p.Add(actualLine, predLine)
	p.Legend.Add("cos(x)", actualLine)
	p.Legend.Add("prediction", predLine)

	p.Save(6*vg.Inch, 4*vg.Inch, "./cmd/cosine/cos_vs_pred.png")
}

func cosine() {
	const samples = 100

	trainSet := nn.Mat{
		Row:     samples,
		Column:  2,
		Weights: make([][]float64, samples),
	}

	for i := 0; i < samples; i++ {
		x := 2 * math.Pi * float64(i) / float64(samples-1)

		trainSet.Weights[i] = []float64{
			x,
			math.Cos(x),
		}
	}

	xor := nn.NewNetwork(
		nn.NewLayer(1, 4, nn.Sigmoid{}),
		nn.NewLayer(4, 4, nn.Sigmoid{}),
		nn.NewLayer(4, 1, nn.Tanh{}),
	).WithOptimizer(nn.Gradient{Rate: 0.1}).WithLoss(nn.MSE{})

	xor.OldTrain(10000, trainSet)

	fmt.Println("===== FINAL PREDICTIONS =====")
	for index := range trainSet.Weights {
		row := trainSet.Weights[index]
		x := row[:len(row)-1]
		pred := xor.Infer(nn.NewRowMat(x))
		fmt.Printf("cos(%v) = %v = |%v\n", x[0], pred, math.Cos(x[0]))
	}
}
