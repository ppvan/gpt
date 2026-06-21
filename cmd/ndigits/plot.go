package main

import (
	"bytes"
	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

// makeCostPlot builds a plot of cost(epoch) = 1/epoch for the epochs
// trained so far, and returns it as encoded PNG bytes. epochs is
// expected to be 1, 2, 3, ... in order (discrete, not continuous).
func makeCostPlot(epochs []float64, costs []float64) ([]byte, error) {
	p := plot.New()
	p.Title.Text = "Training Cost vs. Epoch"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "cost"

	// Always show a sensible range, even before the first click.
	p.X.Min = 0
	p.Y.Min = 0
	if len(epochs) > 0 {
		p.X.Max = epochs[len(epochs)-1] + 1
	} else {
		p.X.Max = 10
	}
	p.Y.Max = 1.1 // cost(1) = 1 is the highest point this function ever reaches

	pts := make(plotter.XYs, len(epochs))
	for i := range epochs {
		pts[i].X = epochs[i]
		pts[i].Y = costs[i]
	}

	if len(pts) > 0 {
		line, err := plotter.NewLine(pts)
		if err != nil {
			return nil, err
		}
		line.Color = color.RGBA{B: 200, A: 255}
		line.Width = vg.Points(1.5)
		p.Add(line)

		scatter, err := plotter.NewScatter(pts)
		if err != nil {
			return nil, err
		}
		scatter.Color = color.RGBA{R: 200, A: 255}
		scatter.Shape = draw.CircleGlyph{}
		scatter.Radius = vg.Points(3)
		p.Add(scatter)

		p.Legend.Add("cost(epoch) = 1/epoch", line)
	}

	canvas := vgimg.New(6*vg.Inch, 4*vg.Inch)
	p.Draw(draw.New(canvas))

	var buf bytes.Buffer
	pngCanvas := vgimg.PngCanvas{Canvas: canvas}
	if _, err := pngCanvas.WriteTo(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
