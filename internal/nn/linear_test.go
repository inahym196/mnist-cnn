package nn

import (
	"math"
	"slices"
	"testing"
)

func TestLinearForward(t *testing.T) {
	l := NewLinear(2, 2)

	l.weights.Value = []float64{
		1, 2,
		3, 4,
	}
	l.biases.Value = []float64{5, 6}

	got := l.Forward([]float64{2, 3})
	want := []float64{13, 24}

	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestLinearForwardNonSquare(t *testing.T) {
	l := NewLinear(2, 3)

	l.weights.Value = []float64{
		1, 0,
		0, 1,
		1, 1,
	}
	l.biases.Value = []float64{0, 0, 0}

	got := l.Forward([]float64{2, 3})
	want := []float64{2, 3, 5}

	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestLinearBackward(t *testing.T) {
	l := NewLinear(2, 2)
	l.weights.Value = []float64{
		1, 2,
		3, 4,
	}
	l.Forward([]float64{2, 3})

	gotInput := l.Backward([]float64{5, 6})

	if want := []float64{23, 34}; !slices.Equal(gotInput, want) {
		t.Errorf("gradInput: got %v, want %v", gotInput, want)
	}

	if want := []float64{10, 15, 12, 18}; !slices.Equal(l.weights.Grad, want) {
		t.Errorf("gradWeights: got %v, want %v", l.weights.Grad, want)
	}

	if want := []float64{5, 6}; !slices.Equal(l.biases.Grad, want) {
		t.Errorf("gradBiases: got %v, want %v", l.biases.Grad, want)
	}
}

func TestLinearGradientAccumulation(t *testing.T) {
	l := NewLinear(2, 1)

	l.Forward([]float64{2, 3})
	l.Backward([]float64{4})

	l.Forward([]float64{1, 5})
	l.Backward([]float64{2})

	if want := []float64{10, 22}; !slices.Equal(l.weights.Grad, want) {
		t.Errorf("gradWeights: got %v, want %v", l.weights.Grad, want)
	}

	if want := []float64{6}; !slices.Equal(l.biases.Grad, want) {
		t.Errorf("gradBiases: got %v, want %v", l.biases.Grad, want)
	}

	l.ZeroGrad()

	if want := []float64{0, 0}; !slices.Equal(l.weights.Grad, want) {
		t.Errorf("gradWeights after reset: got %v", l.weights.Grad)
	}

	if want := []float64{0}; !slices.Equal(l.biases.Grad, want) {
		t.Errorf("gradBiases after reset: got %v", l.biases.Grad)
	}
}

func TestLinearBackwardNumericalGradient(t *testing.T) {
	l := NewLinear(2, 2)
	l.weights.Value = []float64{1, 2, 3, 4}
	l.biases.Value = []float64{7, 8}

	x := []float64{2, 3}
	g := []float64{5, 6}

	l.Forward(x)
	gotInput := l.Backward(g)

	const eps = 1e-5
	const tolerance = 1e-7

	loss := func() float64 {
		y := l.Forward(x)
		return y[0]*g[0] + y[1]*g[1]
	}

	// 重みの勾配
	for i := range l.weights.Value {
		original := l.weights.Value[i]

		l.weights.Value[i] = original + eps
		plus := loss()

		l.weights.Value[i] = original - eps
		minus := loss()

		l.weights.Value[i] = original

		numeric := (plus - minus) / (2 * eps)
		if math.Abs(numeric-l.weights.Grad[i]) > tolerance {
			t.Errorf("weight[%d]: got %v, numeric %v",
				i, l.weights.Grad[i], numeric)
		}
	}

	// バイアスの勾配
	for i := range l.biases.Value {
		original := l.biases.Value[i]

		l.biases.Value[i] = original + eps
		plus := loss()

		l.biases.Value[i] = original - eps
		minus := loss()

		l.biases.Value[i] = original

		numeric := (plus - minus) / (2 * eps)
		if math.Abs(numeric-l.biases.Grad[i]) > tolerance {
			t.Errorf("bias[%d]: got %v, numeric %v",
				i, l.biases.Grad[i], numeric)
		}
	}

	// 入力の勾配
	for i := range x {
		original := x[i]

		x[i] = original + eps
		plus := loss()

		x[i] = original - eps
		minus := loss()

		x[i] = original

		numeric := (plus - minus) / (2 * eps)
		if math.Abs(numeric-gotInput[i]) > tolerance {
			t.Errorf("input[%d]: got %v, numeric %v",
				i, gotInput[i], numeric)
		}
	}
}
