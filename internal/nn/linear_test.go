package nn

import (
	"slices"
	"testing"
)

func TestLinearForward(t *testing.T) {
	l := NewLinear(2, 2)

	l.weights = []float64{
		1, 2,
		3, 4,
	}
	l.biases = []float64{5, 6}

	got := l.Forward([]float64{2, 3})
	want := []float64{13, 24}

	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestLinearForwardNonSquare(t *testing.T) {
	l := NewLinear(2, 3)

	l.weights = []float64{
		1, 0,
		0, 1,
		1, 1,
	}
	l.biases = []float64{0, 0, 0}

	got := l.Forward([]float64{2, 3})
	want := []float64{2, 3, 5}

	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
