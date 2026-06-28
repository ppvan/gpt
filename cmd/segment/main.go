package main

import (
	"fmt"

	"github.com/ppvan/gpt/pkg/txt"
)

func main() {
	corpus, _ := txt.LoadCorpus("./dataset/")
	vocab := txt.BuildVocab(corpus)
	encoded := txt.EncodeCorpus(corpus, vocab)

	vocab.SaveJSON("./dataset/post-processed/vocab.json")

	// Inspect a sentence
	s := encoded[1]
	fmt.Println(s.Text)    // Hắn vừa đi vừa chửi.
	fmt.Println(s.Tokens)  // [hắn vừa đi vừa chửi]
	fmt.Println(s.Encoded) // [2 9 5 8 5 8 3]
	//                         BOS            EOS
}
