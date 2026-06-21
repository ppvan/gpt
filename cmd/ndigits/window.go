package main

import (
	"fmt"

	"github.com/rodrigocfd/windigo/co"
	"github.com/rodrigocfd/windigo/ui"
	"github.com/rodrigocfd/windigo/win"
)

// MyWindow holds the main window, its child controls, and the
// training state accumulated so far.
type MyWindow struct {
	wnd       *ui.Main
	plotCtrl  *ui.Control // blank custom control we paint the plot image onto
	statusLbl *ui.Static
	trainBtn  *ui.Button
	resetBtn  *ui.Button

	epoch       int       // current epoch, starts at 0 (no training yet)
	epochs      []float64 // epoch history: 1, 2, 3, ...
	costs       []float64 // cost(epoch) = 1/epoch history
	plotBmpData []byte    // BMP-encoded plot image, regenerated on each click
}

func ShowMainWindow() int {
	wnd := ui.NewMain(
		ui.OptsMain().
			Title(title).
			Size(ui.Dpi(660, 560)),
	)

	plotCtrl := ui.NewControl(wnd,
		ui.OptsControl().
			Position(ui.Dpi(10, 10)).
			Size(ui.Dpi(620, 440)),
	)

	statusLbl := ui.NewStatic(wnd,
		ui.OptsStatic().
			Text("Epoch: 0    Cost: -").
			Position(ui.Dpi(10, 460)).
			Size(ui.Dpi(400, 23)),
	)

	trainBtn := ui.NewButton(wnd,
		ui.OptsButton().
			Text("Train next epoch").
			Position(ui.Dpi(10, 490)).
			Width(ui.DpiX(200)).
			Height(ui.DpiY(30)),
	)

	resetBtn := ui.NewButton(wnd,
		ui.OptsButton().
			Text("Reset").
			Position(ui.Dpi(220, 490)).
			Width(ui.DpiX(100)).
			Height(ui.DpiY(30)),
	)

	me := &MyWindow{
		wnd:       wnd,
		plotCtrl:  plotCtrl,
		statusLbl: statusLbl,
		trainBtn:  trainBtn,
		resetBtn:  resetBtn,
	}
	me.events()
	return wnd.RunAsMain()
}

func (me *MyWindow) events() {
	// Draw an empty (axes-only) plot as soon as the window is created.
	me.wnd.On().WmCreate(func(_ ui.WmCreate) int {
		me.regeneratePlot()
		return 0
	})

	me.trainBtn.On().BnClicked(func() {
		me.epoch++
		cost := 1.0 / float64(me.epoch)
		me.epochs = append(me.epochs, float64(me.epoch))
		me.costs = append(me.costs, cost)

		me.statusLbl.SetTextAndResize(
			fmt.Sprintf("Epoch: %d    Cost: %.4f", me.epoch, cost),
		)
		me.regeneratePlot()
	})

	me.resetBtn.On().BnClicked(func() {
		me.epoch = 0
		me.epochs = nil
		me.costs = nil
		me.statusLbl.SetTextAndResize("Epoch: 0    Cost: -")
		me.regeneratePlot()
	})

	me.plotCtrl.On().WmPaint(func() {
		var ps win.PAINTSTRUCT
		hdc, err := me.plotCtrl.Hwnd().BeginPaint(&ps)
		if err != nil {
			panic(err)
		}
		defer me.plotCtrl.Hwnd().EndPaint(&ps)

		backgroundBrush, _ := win.GetSysColorBrush(co.COLOR_WINDOW)
		defer backgroundBrush.DeleteObject()
		hdc.FillRect(&ps.RcPaint, backgroundBrush)

		if me.plotBmpData == nil {
			return
		}

		rel := win.NewOleReleaser()
		defer rel.Release() // important: release COM resources to avoid leaks

		stream, err := win.SHCreateMemStream(rel, me.plotBmpData)
		if err != nil {
			panic(err)
		}

		pic, err := win.OleLoadPicture(rel, stream, 0, true)
		if err != nil {
			panic(err)
		}

		sz, _ := pic.Size()
		_, _ = pic.Render(hdc,
			win.POINT{},
			win.SIZE{Cx: ps.RcPaint.Right, Cy: ps.RcPaint.Bottom},
			win.POINT{X: 0, Y: sz.Cy},
			win.SIZE{Cx: sz.Cx, Cy: -sz.Cy},
		)
	})
}

// regeneratePlot re-renders the cost(epoch) plot from the current
// epoch/cost history and triggers a repaint of the image control.
func (me *MyWindow) regeneratePlot() {
	pngData, err := makeCostPlot(me.epochs, me.costs)
	if err != nil {
		panic(err)
	}
	bmpData, err := pngToBitmapInMemory(pngData)
	if err != nil {
		panic(err)
	}
	me.plotBmpData = bmpData
	me.plotCtrl.Hwnd().RedrawWindow(nil, 0, co.RDW_INVALIDATE)
}
