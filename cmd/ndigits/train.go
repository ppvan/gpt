package main

import (
	"github.com/ppvan/gpt/nn"
)

// digitsCsvPath is the location of the training data, relative to the
// working directory the .exe is run from (matches the reference snippet).
const digitsCsvPath = "./data/digits.csv"

// trainResult is handed back to the UI thread once training finishes.
type trainResult struct {
	finalLoss float64
	err       error
}

// runTraining loads the digits dataset, builds the network, and trains it
// for the given number of epochs. onEpoch is called after every epoch
// completes; it may be called from a background goroutine, so callers
// must marshal any UI work back to the UI thread themselves (e.g. via
// (*ui.Main).UiThread).
//
// This function blocks until training finishes (or fails to load data),
// so it should always be called from inside a goroutine, never directly
// from a windigo event handler.
func runTraining(epochs int, onEpoch func(epoch int, loss float64)) trainResult {
	raw, err := nn.LoadCSV(digitsCsvPath, 1, true)
	if err != nil {
		return trainResult{err: err}
	}

	y := raw.Y.OneHot(10) // 1796x1 labels -> 1796x10 one-hot
	data := nn.NewDataset(raw.X, y)

	ndigit := nn.NewNetwork([]int{64, 32, 10})
	result := ndigit.Train(epochs, data, nn.TrainConfig{
		BatchSize: 100,
		OnEpoch:   onEpoch,
	})

	finalLoss := result.EpochLosses[len(result.EpochLosses)-1]
	return trainResult{finalLoss: finalLoss}
}
