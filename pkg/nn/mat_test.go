package nn

import (
	"testing"
)

func benchmarkMultiply(b *testing.B, m, n, p int) {
	a := randomMat(m, n)
	c := randomMat(n, p)

	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		_ = a.Multiply(c)
	}
}

func BenchmarkMultiply32x32(b *testing.B) {
	benchmarkMultiply(b, 32, 32, 32)
}

func BenchmarkMultiply64x64(b *testing.B) {
	benchmarkMultiply(b, 64, 64, 64)
}

func BenchmarkMultiply128x128(b *testing.B) {
	benchmarkMultiply(b, 128, 128, 128)
}

func BenchmarkMultiplyBatch32_64x32(b *testing.B) {
	benchmarkMultiply(b, 32, 64, 32)
}

func BenchmarkMultiplyBatch32_32x64(b *testing.B) {
	benchmarkMultiply(b, 32, 32, 64)
}

func BenchmarkMultiplyBatch32_64x10(b *testing.B) {
	benchmarkMultiply(b, 32, 64, 10)
}

func BenchmarkMultiplyBatch32_10x10(b *testing.B) {
	benchmarkMultiply(b, 32, 10, 10)
}
