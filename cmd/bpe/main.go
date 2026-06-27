package main

import (
	"fmt"
	"os"

	"github.com/ppvan/gpt/bpe"
)

func main() {

	bytes, err := os.ReadFile("./cmd/bpe/lao_hac_nam_cao.txt")
	if err != nil {
		panic(err)
	}

	text := string(bytes)

	t := bpe.New()
	if err := t.Train(text, 1024); err != nil {
		fmt.Println("Train error:", err)
		return
	}

	vocab := t.Vocab()
	fmt.Println("--- Learned Vocab (merged tokens) ---")
	for id := 256; id < len(vocab); id++ {
		fmt.Printf("  vocab[%d] = %q\n", id, string(vocab[id]))
	}

	samples := []string{
		"lão hạc",
	}

	fmt.Println("\n--- Encode/Decode round-trips ---")
	for _, sample := range samples {
		encoded := t.Encode(sample)
		decoded, err := t.Decode(encoded)
		if err != nil {
			fmt.Printf("Decode error: %v\n", err)
			continue
		}
		fmt.Printf("Original : %q\n", sample)
		fmt.Printf("Encoded  : %v\n", encoded)
		fmt.Printf("Decoded  : %q\n", decoded)
		fmt.Printf("Match    : %v\n\n", sample == decoded)
	}
}
