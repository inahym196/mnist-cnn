package optim

import (
	"math"
	"testing"

	"github.com/inahym196/mnist-cnn/internal/nn"
)

func TestSGDStep(t *testing.T) {
	p := nn.Parameter{
		Value: []float64{10, -5},
		Grad:  []float64{4, -2},
	}

	sgd := NewSGD(0.1)
	sgd.Step(&p)

	want := []float64{9.6, -4.8}

	for i, got := range p.Value {
		if math.Abs(got-want[i]) > 1e-12 {
			t.Errorf("value[%d]: got %v, want %v", i, got, want[i])
		}
	}
}

func TestSGDPreservesGradient(t *testing.T) {
	p := nn.Parameter{
		Value: []float64{10},
		Grad:  []float64{4},
	}

	NewSGD(0.1).Step(&p)

	if p.Grad[0] != 4 {
		t.Fatalf("gradient was modified: %v", p.Grad[0])
	}
}
