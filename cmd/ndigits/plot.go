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

var (
	lineColor   = color.RGBA{R: 0, G: 90, B: 200, A: 255}
	markerColor = color.RGBA{R: 200, G: 40, B: 0, A: 255}
	gridColor   = color.RGBA{R: 220, G: 220, B: 220, A: 255}
)

// makeLossPlot builds a plot of training loss vs. epoch for the epochs
// trained so far, and returns it as encoded PNG bytes. epochs is
// expected to be 1, 2, 3, ... in order (discrete, not continuous).
func makeLossPlot(epochs []float64, losses []float64) ([]byte, error) {
	p := plot.New()
	p.BackgroundColor = color.White
	p.Title.Text = "Training Loss vs. Epoch"
	p.Title.Padding = vg.Points(8)
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "loss"
	p.X.Padding = vg.Points(5)
	p.Y.Padding = vg.Points(5)

	// Light gridlines, added first so the loss line draws on top of them.
	grid := plotter.NewGrid()
	grid.Vertical.Color = gridColor
	grid.Vertical.Width = vg.Points(0.5)
	grid.Horizontal.Color = gridColor
	grid.Horizontal.Width = vg.Points(0.5)
	p.Add(grid)

	// Pin the origin so the chart doesn't visually jump around as points
	// are added; Plot.Add only ever grows this range, never shrinks it,
	// so real loss values (whatever their scale) still fit in.
	p.X.Min = 0
	p.Y.Min = 0
	if len(epochs) > 0 {
		p.X.Max = epochs[len(epochs)-1] + 1
	} else {
		p.X.Max = 10
	}

	pts := make(plotter.XYs, len(epochs))
	for i := range epochs {
		pts[i].X = epochs[i]
		pts[i].Y = losses[i]
	}

	if len(pts) > 0 {
		line, err := plotter.NewLine(pts)
		if err != nil {
			return nil, err
		}
		line.Color = lineColor
		line.Width = vg.Points(1.5)
		p.Add(line)

		// Only draw individual markers for a manageable number of points;
		// with hundreds of epochs the markers just clutter the line.
		if len(pts) <= 100 {
			scatter, err := plotter.NewScatter(pts)
			if err != nil {
				return nil, err
			}
			scatter.Color = markerColor
			scatter.Shape = draw.CircleGlyph{}
			scatter.Radius = vg.Points(2.5)
			p.Add(scatter)
		}

		p.Legend.Add("loss", line)
	}

	// Anchor the legend to a fixed corner of the plot area (top-right),
	// instead of the default bottom-right, which collides with the line
	// once the loss curve flattens out near y=0.
	p.Legend.Top = true
	p.Legend.Left = false
	p.Legend.XOffs = -vg.Points(8)
	p.Legend.YOffs = -vg.Points(8)
	p.Legend.ThumbnailWidth = vg.Points(20)
	p.Legend.Padding = vg.Points(4)

	canvas := vgimg.New(6*vg.Inch, 4*vg.Inch)
	p.Draw(draw.New(canvas))

	var buf bytes.Buffer
	pngCanvas := vgimg.PngCanvas{Canvas: canvas}
	if _, err := pngCanvas.WriteTo(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
