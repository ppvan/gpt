// Package bpe implements a byte-pair encoding tokenizer.
//
// Usage:
//
//	t := bpe.New()
//	t.Train(text, 276)
//	ids  := t.Encode("hello world")
//	text := t.Decode(ids)
//	vocab := t.Vocab()
package bpe

import (
	"fmt"
)

type mergeRule struct {
	a, b  int
	newID int
}

type Tokenizer struct {
	vocab      [][]byte
	mergeRules []mergeRule
	mergeIndex map[[2]int]int
}

func New() *Tokenizer {
	vocab := make([][]byte, 256)
	for i := range 256 {
		vocab[i] = []byte{byte(i)}
	}
	return &Tokenizer{
		vocab:      vocab,
		mergeIndex: make(map[[2]int]int),
	}
}

func (t *Tokenizer) Vocab() [][]byte {
	out := make([][]byte, len(t.vocab))
	for i, token := range t.vocab {
		cp := make([]byte, len(token))
		copy(cp, token)
		out[i] = cp
	}
	return out
}

func (t *Tokenizer) Train(text string, vocabSize int) error {
	if vocabSize <= 256 {
		return fmt.Errorf("vocabSize must be > 256, got %d", vocabSize)
	}

	numMerges := vocabSize - 256
	ids := bytesToIDs([]byte(text))

	for step := range numMerges {
		counts := countPairs(ids)
		if len(counts) == 0 {
			break // corpus too small to produce more merges
		}

		pair := mostFrequent(counts)
		a, b := pair[0], pair[1]
		newToken := concat(t.vocab[a], t.vocab[b])
		newID := t.addToken(newToken)

		ids = applyMerge(ids, a, b, newID)

		rule := mergeRule{a, b, newID}
		t.mergeRules = append(t.mergeRules, rule)
		t.mergeIndex[pair] = newID

		_ = step // step available here if callers want progress logging
	}

	return nil
}

func (t *Tokenizer) Encode(text string) []int {
	ids := bytesToIDs([]byte(text))
	for _, rule := range t.mergeRules {
		ids = applyMerge(ids, rule.a, rule.b, rule.newID)
	}
	return ids
}

func (t *Tokenizer) Decode(ids []int) (string, error) {
	var buf []byte
	for _, id := range ids {
		if id < 0 || id >= len(t.vocab) {
			return "", fmt.Errorf("unknown token ID %d", id)
		}
		buf = append(buf, t.vocab[id]...)
	}
	return string(buf), nil
}

// ---- internal helpers ----

func (t *Tokenizer) addToken(data []byte) int {
	id := len(t.vocab)
	t.vocab = append(t.vocab, data)
	return id
}

func bytesToIDs(b []byte) []int {
	ids := make([]int, len(b))
	for i, v := range b {
		ids[i] = int(v)
	}
	return ids
}

func countPairs(ids []int) map[[2]int]int {
	counts := make(map[[2]int]int)
	for i := range len(ids) - 1 {
		counts[[2]int{ids[i], ids[i+1]}]++
	}
	return counts
}

func mostFrequent(counts map[[2]int]int) [2]int {
	best := [2]int{}
	bestCount := 0
	for pair, count := range counts {
		if count > bestCount {
			bestCount = count
			best = pair
		}
	}
	return best
}

func applyMerge(ids []int, a, b, newID int) []int {
	out := make([]int, 0, len(ids))
	i := 0
	for i < len(ids) {
		if i < len(ids)-1 && ids[i] == a && ids[i+1] == b {
			out = append(out, newID)
			i += 2
		} else {
			out = append(out, ids[i])
			i++
		}
	}
	return out
}

func concat(a, b []byte) []byte {
	out := make([]byte, len(a)+len(b))
	copy(out, a)
	copy(out[len(a):], b)
	return out
}
