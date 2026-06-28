package main

import (
	"fmt"
	"math/rand"

	"github.com/ppvan/gpt/pkg/txt"
)

func main() {
	corpus, _ := txt.LoadCorpus("./dataset/")
	vocab := corpus.BuildVocab()

	fmt.Println(corpus)

	encoded := txt.EncodeCorpus(corpus, vocab)

	// Inspect a sentence
	s := encoded[int(rand.Float32()*float32(len(encoded)))]
	fmt.Println(s.Text)    // Hắn vừa đi vừa chửi.
	fmt.Println(s.Tokens)  // [hắn vừa đi vừa chửi]
	fmt.Println(s.Encoded) // [2 9 5 8 5 8 3]
	//                         BOS            EOS

	vocab.SaveJSON("./dataset/post-processed/vocab.json")
}
