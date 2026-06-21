package main

import (
	"runtime"

	"github.com/rodrigocfd/windigo/co"
	"github.com/rodrigocfd/windigo/win"
)

const title = "Cosine Plot"

func main() {
	runtime.LockOSThread() // important: Windows GUI must run on a single OS thread

	win.CoInitializeEx(co.COINIT_APARTMENTTHREADED | co.COINIT_DISABLE_OLE1DDE)
	defer win.CoUninitialize()

	ShowMainWindow()
}
