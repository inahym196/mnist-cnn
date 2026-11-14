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

func TestSDGWithLinear(t *testing.T) {
	l := nn.NewLinear(1, 1)
	sgd := NewSGD(0.1)

	x := []float64{1}

	loss := func(y float64) float64 {
		return 0.5 * math.Pow(y-1.0, 2)
	}

	gradLoss := func(y float64) float64 {
		return y - 1.0
	}

	y := l.Forward(x)[0]
	before := loss(y)

	l.Backward([]float64{gradLoss(y)})
	sgd.Step(l.Parameters()...)

	y = l.Forward(x)[0]
	after := loss(y)

	if math.Abs(after-0.32) > 1e-12 {
		t.Fatalf("unexpected loss: got %v, want 0.32", after)
	}

	if after >= before {
		t.Fatalf("loss did not decrease: before=%v, after=%v", before, after)
	}
}
