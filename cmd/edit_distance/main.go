package main

import (
	"fmt"
)

func main() {

	str := "kitten"
	other := "sitting"

	fmt.Printf("Distance lev(%q, %q): %d\n", str, other, lev(str, other))

}

func lev(a string, b string) int {

	rows := len(a) + 1
	columns := len(b) + 1
	d := make([]int, rows*columns)

	for i := range rows {
		d[columns*i] = i
	}

	for j := range columns {
		d[j] = j
	}

	for i := 1; i < rows; i++ {
		for j := 1; j < columns; j++ {
			sub := 0
			if a[i-1] == b[j-1] {
				sub = 0
			} else {
				sub = 1
			}

			d[columns*(i)+j] = min(
				d[columns*(i-1)+j]+1,
				min(
					d[columns*i+j-1]+1,
					d[columns*(i-1)+j-1]+sub,
				),
			)
		}
	}

	return d[rows*columns-1]

}
