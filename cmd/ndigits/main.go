package main

//go:generate go-winres make --product-version=git-tag

import (
	"runtime"

	"github.com/rodrigocfd/windigo/co"
	"github.com/rodrigocfd/windigo/win"
)

const title = "Digit Recognizer Training"

func main() {
	runtime.LockOSThread() // important: Windows GUI must run on a single OS thread

	win.CoInitializeEx(co.COINIT_APARTMENTTHREADED | co.COINIT_DISABLE_OLE1DDE)
	defer win.CoUninitialize()

	ShowMainWindow()
}
