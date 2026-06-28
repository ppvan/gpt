package main

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
)

type Document struct {
	paragraghs []string
}

func main() {
	path := "dataset/Nam Cao/Chí Phèo.txt"
	handle, err := os.Open(path)
	if err != nil {
		panic(err)
	}

	content, err := io.ReadAll(handle)
	if err != nil {
		panic(err)
	}

	text := string(content)
	lines := strings.Split(text, "\n\n")
	stops := regexp.MustCompile(`(?m)(\.|\?|!) `)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "***" {
			continue
		}

		sentences := stops.Split(line, 1)
		for _, sen := range sentences {
			fmt.Printf("%v\n", sen)
			fmt.Println("-------------------------------------------------------------")
		}

	}
}
