package nn

import (
	"slices"
	"testing"
)

func TestReLUForward(t *testing.T) {
	r := NewReLU()

	got := r.Forward([]float64{-2, 0, 3, 5})
	want := []float64{0, 0, 3, 5}

	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestReLUBackward(t *testing.T) {
	r := NewReLU()

	r.Forward([]float64{-2, 0, 3, 5})

	got := r.Backward([]float64{1, 2, 3, 4})
	want := []float64{0, 0, 3, 4}

	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
