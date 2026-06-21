package main

import (
	"fmt"

	"github.com/rodrigocfd/windigo/co"
	"github.com/rodrigocfd/windigo/ui"
	"github.com/rodrigocfd/windigo/win"
)

// trainEpochs is how many epochs runTraining is asked to run for.
const trainEpochs = 1000

// redrawEveryN throttles how often we regenerate and repaint the plot
// image while training is running. Re-rendering a 6x4in PNG on every
// single epoch of a 1000-epoch run would be wasteful and would also
// flood the UI thread with work; we redraw, at most, every N epochs.
const redrawEveryN = 10

// MyWindow holds the main window, its child controls, and the
// training state accumulated so far.
type MyWindow struct {
	wnd       *ui.Main
	plotCtrl  *ui.Control // blank custom control we paint the plot image onto
	statusLbl *ui.Static
	trainBtn  *ui.Button

	epochs      []float64 // epoch history: 1, 2, 3, ...
	losses      []float64 // loss history, one entry per epoch
	plotBmpData []byte    // BMP-encoded plot image, regenerated as training progresses
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
			Text("Epoch: 0    Loss: -").
			Position(ui.Dpi(10, 460)).
			Size(ui.Dpi(500, 23)),
	)

	trainBtn := ui.NewButton(wnd,
		ui.OptsButton().
			Text("Start Training").
			Position(ui.Dpi(10, 490)).
			Width(ui.DpiX(200)).
			Height(ui.DpiY(30)),
	)

	me := &MyWindow{
		wnd:       wnd,
		plotCtrl:  plotCtrl,
		statusLbl: statusLbl,
		trainBtn:  trainBtn,
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
		// Reset state for a fresh run.
		me.epochs = nil
		me.losses = nil
		me.statusLbl.SetTextAndResize("Epoch: 0    Loss: -")
		me.regeneratePlot()

		me.trainBtn.Hwnd().EnableWindow(false)
		me.trainBtn.SetText("Training...")

		// Training is a blocking, potentially long-running call, so it
		// must run off the UI thread. Every UI touch from inside the
		// goroutine (including from onEpoch) must be marshaled back via
		// wnd.UiThread, since windigo/Win32 GUI calls are not safe to
		// make from any thread other than the one that owns the window.
		go func() {
			result := runTraining(trainEpochs, func(epoch int, loss float64) {
				me.wnd.UiThread(func() {
					me.epochs = append(me.epochs, float64(epoch))
					me.losses = append(me.losses, loss)

					me.statusLbl.SetTextAndResize(
						fmt.Sprintf("Epoch: %d    Loss: %.6f", epoch, loss),
					)

					if epoch%redrawEveryN == 0 {
						me.regeneratePlot()
					}
				})
			})

			me.wnd.UiThread(func() {
				me.trainBtn.SetText("Start Training")
				me.trainBtn.Hwnd().EnableWindow(true)

				if result.err != nil {
					me.statusLbl.SetTextAndResize("Training failed: " + result.err.Error())
					return
				}

				me.statusLbl.SetTextAndResize(
					fmt.Sprintf("Done. Final loss: %.6f", result.finalLoss),
				)
				me.regeneratePlot() // make sure the very last epoch is on screen
			})
		}()
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

func (me *MyWindow) regeneratePlot() {
	pngData, err := makeLossPlot(me.epochs, me.losses)
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
