package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"

	chart "github.com/wcharczuk/go-chart/v2"
	"github.com/wcharczuk/go-chart/v2/drawing"
)

var (
	lineColor = drawing.Color{R: 0, G: 90, B: 200, A: 255}
	gridColor = drawing.Color{R: 220, G: 220, B: 220, A: 255}
)

func makeLossPlot(epochs []float64, losses []float64) ([]byte, error) {
	if len(epochs) < 2 {
		return blankPNG(864, 576)
	}

	lineSeries := chart.ContinuousSeries{
		Name:    "loss",
		XValues: epochs,
		YValues: losses,
		YAxis:   chart.YAxisPrimary,
		Style: chart.Style{
			StrokeColor: lineColor,
			StrokeWidth: 2,
			DotWidth:    0,
		},
	}

	series := []chart.Series{lineSeries}

	xMax := epochs[len(epochs)-1] + 1
	yMin, yMax := minMax(losses)

	c := chart.Chart{
		DPI:    144,
		Width:  1296,
		Height: 864,
		Background: chart.Style{
			FillColor: drawing.Color{R: 255, G: 255, B: 255, A: 255},
		},
		XAxis: chart.XAxis{
			Name:  "epoch",
			Ticks: makeEpochTicks(epochs),
			Range: &chart.ContinuousRange{
				Min: 0,
				Max: xMax,
			},
			GridMajorStyle: chart.Style{
				StrokeColor: gridColor,
				StrokeWidth: 0.5,
			},
			GridLines: []chart.GridLine{},
		},
		YAxis: chart.YAxis{
			Name:  "loss",
			Ticks: makeLossTicks(yMin, yMax),
			Range: &chart.ContinuousRange{Min: 0, Max: yMax * 1.1},
			GridMajorStyle: chart.Style{
				StrokeColor: gridColor,
				StrokeWidth: 0.5,
			},
			GridLines: []chart.GridLine{},
		},
		Series: series,
	}

	c.Elements = []chart.Renderable{
		chart.LegendThin(&c),
	}

	var buf bytes.Buffer
	if err := c.Render(chart.PNG, &buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func blankPNG(width, height int) ([]byte, error) {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	white := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, white)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func makeEpochTicks(epochs []float64) []chart.Tick {
	if len(epochs) == 0 {
		return nil
	}

	maxEpoch := epochs[len(epochs)-1]
	step := niceStep(maxEpoch, 10)

	var ticks []chart.Tick
	for v := 0.0; v <= maxEpoch+step; v += step {
		ticks = append(ticks, chart.Tick{
			Value: v,
			Label: fmt.Sprintf("%.0f", v),
		})
	}
	return ticks
}

func minMax(vals []float64) (float64, float64) {
	mn, mx := vals[0], vals[0]
	for _, v := range vals[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return mn, mx
}

func niceStep(maxVal float64, maxTicks int) float64 {
	rough := maxVal / float64(maxTicks)

	mag := math.Pow(10, math.Floor(math.Log10(rough)))

	for _, mult := range []float64{1, 2, 5, 10} {
		step := mag * mult
		if maxVal/step <= float64(maxTicks) {
			return step
		}
	}
	return mag * 10
}

func makeLossTicks(yMin, yMax float64) []chart.Tick {
	if yMax <= 0 {
		return nil
	}

	magMax := math.Floor(math.Log10(yMax))                      // e.g.  0 for yMax=1.0
	magMin := math.Floor(math.Log10(math.Max(yMin, yMax*1e-6))) // e.g. -3 for 0.001

	decades := magMax - magMin

	if decades >= 2 {
		return logTicks(magMin, magMax)
	}
	return linearTicks(0, yMax, 8)
}

func logTicks(magMin, magMax float64) []chart.Tick {
	decades := int(magMax - magMin)

	addMid := decades <= 3

	var ticks []chart.Tick
	for mag := magMax; mag >= magMin; mag-- {
		major := math.Pow(10, mag)
		ticks = append(ticks, chart.Tick{
			Value: major,
			Label: formatLoss(major),
		})
		if addMid {
			mid := major / 2 // 0.5, 0.05, 0.005 ...
			if mid > 0 {
				ticks = append(ticks, chart.Tick{
					Value: mid,
					Label: formatLoss(mid),
				})
			}
		}
	}

	ticks = append(ticks, chart.Tick{Value: 0, Label: "0"})
	return ticks
}

func linearTicks(min, max float64, maxTicks int) []chart.Tick {
	step := niceStep(max-min, maxTicks)
	var ticks []chart.Tick
	ticks = append(ticks, chart.Tick{Value: 0, Label: "0"})
	for v := step; v <= max+step/2; v += step {
		ticks = append(ticks, chart.Tick{
			Value: v,
			Label: formatLoss(v),
		})
	}
	return ticks
}

func formatLoss(v float64) string {
	switch {
	case v == 0:
		return "0"
	case v >= 0.01:
		return fmt.Sprintf("%.2f", v)
	default:
		return ""
	}
}
