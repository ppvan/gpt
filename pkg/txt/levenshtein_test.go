package txt

import "testing"

func TestLevenshtein(t *testing.T) {
	tests := []struct {
		name string // description of this test case
		// Named input parameters for target function.
		a    string
		b    string
		want int
	}{
		{
			name: "normal",
			a:    "kitten",
			b:    "sitting",
			want: 3,
		},
		{
			name: "delete",
			a:    "kitten",
			b:    "",
			want: 6,
		},
		{
			name: "adding",
			a:    "hello",
			b:    "hello world",
			want: 6,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Levenshtein(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("Levenshtein() = %v, want %v", got, tt.want)
			}
		})
	}
}
