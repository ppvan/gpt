package txt

import (
	"encoding/json"
	"fmt"
	"os"
)

type Vocab struct {
	Token2ID map[string]int
	ID2Token map[int]string
}

func (v *Vocab) Size() int {
	return len(v.Token2ID)
}

func (v *Vocab) Encode(token string) int {
	if id, ok := v.Token2ID[token]; ok {
		return id
	}
	return UNK_ID
}

func (v *Vocab) SaveJSON(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")

	return enc.Encode(v.Token2ID)
}

func LoadVocab(path string) (Vocab, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Vocab{}, fmt.Errorf("read vocab %q: %w", path, err)
	}

	token2id := make(map[string]int)
	if err := json.Unmarshal(data, &token2id); err != nil {
		return Vocab{}, fmt.Errorf("parse vocab: %w", err)
	}

	id2token := make(map[int]string, len(token2id))
	for token, id := range token2id {
		id2token[id] = token
	}

	return Vocab{Token2ID: token2id, ID2Token: id2token}, nil
}
