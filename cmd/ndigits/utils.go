package main

import (
	"bytes"
	"fmt"
	"image"

	"golang.org/x/image/bmp"
)

// pngToBitmapInMemory converts PNG bytes to BMP bytes in memory.
// win.OleLoadPicture via GDI+ chokes on some PNGs in this code path,
// so we follow the same approach as the vye example: decode then
// re-encode as BMP before handing it to the paint handler.
func pngToBitmapInMemory(pngData []byte) ([]byte, error) {
	img, _, err := image.Decode(bytes.NewReader(pngData))
	if err != nil {
		return nil, fmt.Errorf("failed to decode PNG: %w", err)
	}

	var bmpBuffer bytes.Buffer
	if err := bmp.Encode(&bmpBuffer, img); err != nil {
		return nil, fmt.Errorf("failed to encode BMP: %w", err)
	}
	return bmpBuffer.Bytes(), nil
}
