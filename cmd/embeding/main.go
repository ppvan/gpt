package main

import (
	"fmt"
	"math"
	"os"

	"github.com/ppvan/gpt/pkg/nn"
	"github.com/ppvan/gpt/pkg/txt"
)

const embeddingDim = 8 // small for readable inspection

func main() {
	// ── Stage 1: Load corpus ─────────────────────────────────────────────
	corpus, err := txt.LoadCorpus("dataset")
	if err != nil {
		fmt.Fprintf(os.Stderr, "load corpus: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(corpus.String())

	// ── Stage 2: Build vocab ─────────────────────────────────────────────
	vocab := corpus.BuildVocab()
	fmt.Printf("\nVocab size: %d\n", vocab.Size())

	// ── Stage 3: Encode a sample sentence ────────────────────────────────
	sample := corpus.Sentences[0]
	tokens := txt.Tokenize(sample.Text)
	encoded := vocab.EncodeTokens(tokens)

	fmt.Printf("\n── Sample sentence ──────────────────────────────────\n")
	fmt.Printf("Source  : %s\n", sample.Source)
	fmt.Printf("Text    : %s\n", sample.Text)
	fmt.Printf("Tokens  : %v\n", tokens)
	fmt.Printf("Encoded : %v\n", encoded)

	// ── Stage 2: Embedding layer ─────────────────────────────────────────
	embed := nn.NewEmbedding(vocab.Size(), embeddingDim)

	// Build input Mat [seqLen × 1] from encoded IDs (skip BOS/EOS for clarity)
	ids := encoded[1 : len(encoded)-1] // strip BOS, EOS
	input := toIDMat(ids)

	output := embed.Forward(input)

	// ── Inspect ───────────────────────────────────────────────────────────
	fmt.Printf("\n── Embedding output [%d tokens × %d dims] ───────────\n",
		output.Rows, output.Columns)

	for i, token := range tokens {
		vec := output.RowAt(i)
		fmt.Printf("  %-12s → %s  (norm=%.3f)\n",
			token, formatVec(vec), norm(vec))
	}

	// ── Similarity spot-check ─────────────────────────────────────────────
	// Before training, embeddings are random — similarity should be ~0
	// This gives us a baseline to compare after training
	if len(tokens) >= 2 {
		v1 := output.RowAt(0)
		v2 := output.RowAt(1)
		fmt.Printf("\n── Cosine similarity (%s, %s) before training: %.4f\n",
			tokens[0], tokens[1], cosine(v1, v2))
		fmt.Println("  (random init → near 0, after training similar words → near 1)")
	}
}

// toIDMat converts []int token IDs to a Mat [n × 1].
func toIDMat(ids []int) nn.Mat {
	rows := make([][]float64, len(ids))
	for i, id := range ids {
		rows[i] = []float64{float64(id)}
	}
	return nn.NewMat(rows)
}

// formatVec prints a float64 slice as fixed-width decimals.
func formatVec(v []float64) string {
	s := "["
	for i, x := range v {
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("%6.3f", x)
	}
	return s + "]"
}

// norm computes the L2 norm of a vector.
func norm(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// cosine computes cosine similarity between two vectors.
func cosine(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
