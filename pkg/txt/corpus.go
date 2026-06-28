package txt

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

type Sentence struct {
	Text   string
	Source string
}

type fileStats struct {
	source    string
	sentences int
}

type Corpus struct {
	Sentences []Sentence
	files     []fileStats
}

const (
	PAD = "<PAD>"
	UNK = "<UNK>"
	BOS = "<BOS>"
	EOS = "<EOS>"

	PAD_ID = 0
	UNK_ID = 1
	BOS_ID = 2
	EOS_ID = 3

	MinFreq = 2
)

func (c *Corpus) TotalSentences() int {
	return len(c.Sentences)
}

func (c *Corpus) String() string {
	return fmt.Sprintf("Corpus { sentences: %d, files: %d }",
		len(c.Sentences), len(c.files))
}

// LoadCorpus walks dir recursively, loading all .txt files found.
func LoadCorpus(dir string) (*Corpus, error) {
	var corpus Corpus

	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return fmt.Errorf("walk %q: %w", path, err)
		}
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".txt") {
			return nil
		}

		// Use path relative to root dir as source to reserves author/title context
		// e.g. "Nam Cao/Chí Phèo" instead of just "Chí Phèo"
		rel, _ := filepath.Rel(dir, path)
		source := strings.TrimSuffix(rel, ".txt")

		sentences, err := loadSentencesFromFile(path)
		if err != nil {
			return fmt.Errorf("load %q: %w", path, err)
		}

		for _, text := range sentences {
			corpus.Sentences = append(corpus.Sentences, Sentence{
				Text:   text,
				Source: source,
			})
		}
		corpus.files = append(corpus.files, fileStats{source: source, sentences: len(sentences)})
		return nil
	})

	if err != nil {
		return &Corpus{}, err
	}
	return &corpus, nil
}

func LoadDocument(path string) (Corpus, error) {
	source := strings.TrimSuffix(filepath.Base(path), ".txt")
	sentences, err := loadSentencesFromFile(path)
	if err != nil {
		return Corpus{}, err
	}

	var corpus Corpus
	for _, text := range sentences {
		corpus.Sentences = append(corpus.Sentences, Sentence{
			Text:   text,
			Source: source,
		})
	}
	corpus.files = append(corpus.files, fileStats{source: source, sentences: len(sentences)})
	return corpus, nil
}

var sentenceEnds = regexp.MustCompile(`([.?!])\s+`)

func loadSentencesFromFile(path string) ([]string, error) {
	handle, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %q: %w", path, err)
	}
	defer handle.Close()

	content, err := io.ReadAll(handle)
	if err != nil {
		return nil, fmt.Errorf("read %q: %w", path, err)
	}
	return extractSentences(string(content)), nil
}

func extractSentences(text string) []string {
	var sentences []string

	paragraphs := strings.Split(text, "\n\n")
	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" || para == "***" {
			continue
		}

		parts := sentenceEnds.Split(para, -1)
		puncts := sentenceEnds.FindAllStringSubmatch(para, -1)

		for i, part := range parts {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			if i < len(puncts) {
				part = part + puncts[i][1]
			}
			sentences = append(sentences, part)
		}
	}
	return sentences
}

func Tokenize(text string) []string {
	text = strings.ToLower(text)
	raw := strings.Fields(text)

	tokens := make([]string, 0, len(raw))
	for _, word := range raw {
		cleaned := strings.TrimFunc(word, func(r rune) bool {
			return unicode.IsPunct(r) || unicode.IsSymbol(r)
		})
		if cleaned != "" {
			tokens = append(tokens, cleaned)
		}
	}
	return tokens
}

func (corpus *Corpus) BuildVocab() *Vocab {
	freq := make(map[string]int)
	for _, s := range corpus.Sentences {
		for _, token := range Tokenize(s.Text) {
			freq[token]++
		}
	}

	qualified := make([]string, 0, len(freq))
	for token, count := range freq {
		if count >= MinFreq {
			qualified = append(qualified, token)
		}
	}
	sort.Strings(qualified)

	token2id := map[string]int{
		PAD: PAD_ID,
		UNK: UNK_ID,
		BOS: BOS_ID,
		EOS: EOS_ID,
	}
	id2token := map[int]string{
		PAD_ID: PAD,
		UNK_ID: UNK,
		BOS_ID: BOS,
		EOS_ID: EOS,
	}

	nextID := 4
	for _, token := range qualified {
		token2id[token] = nextID
		id2token[nextID] = token
		nextID++
	}

	return &Vocab{Token2ID: token2id, ID2Token: id2token}
}

// TokenizedSentence holds the original sentence alongside its token and encoded forms.
type TokenizedSentence struct {
	Source  string
	Text    string
	Tokens  []string
	Encoded []int
}

// Encode converts a token slice to integer IDs, wrapped with BOS/EOS.
func (v *Vocab) EncodeTokens(tokens []string) []int {
	ids := make([]int, 0, len(tokens)+2)
	ids = append(ids, BOS_ID)
	for _, token := range tokens {
		ids = append(ids, v.Encode(token))
	}
	ids = append(ids, EOS_ID)
	return ids
}

// EncodeCorpus tokenizes and encodes every sentence in the corpus.
func EncodeCorpus(corpus *Corpus, vocab *Vocab) []TokenizedSentence {
	result := make([]TokenizedSentence, 0, len(corpus.Sentences))
	for _, s := range corpus.Sentences {
		tokens := Tokenize(s.Text)
		if len(tokens) == 0 {
			continue
		}
		result = append(result, TokenizedSentence{
			Source:  s.Source,
			Text:    s.Text,
			Tokens:  tokens,
			Encoded: vocab.EncodeTokens(tokens),
		})
	}
	return result
}
