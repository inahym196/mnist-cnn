package nn

import (
	"math"
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

func TestLinearBackward(t *testing.T) {
	l := NewLinear(2, 2)
	l.weights = []float64{
		1, 2,
		3, 4,
	}
	l.Forward([]float64{2, 3})

	gotInput := l.Backward([]float64{5, 6})

	if want := []float64{23, 34}; !slices.Equal(gotInput, want) {
		t.Errorf("gradInput: got %v, want %v", gotInput, want)
	}

	if want := []float64{10, 15, 12, 18}; !slices.Equal(l.gradWeights, want) {
		t.Errorf("gradWeights: got %v, want %v", l.gradWeights, want)
	}

	if want := []float64{5, 6}; !slices.Equal(l.gradBiases, want) {
		t.Errorf("gradBiases: got %v, want %v", l.gradBiases, want)
	}
}

func TestLinearGradientAccumulation(t *testing.T) {
	l := NewLinear(2, 1)

	l.Forward([]float64{2, 3})
	l.Backward([]float64{4})

	l.Forward([]float64{1, 5})
	l.Backward([]float64{2})

	if want := []float64{10, 22}; !slices.Equal(l.gradWeights, want) {
		t.Errorf("gradWeights: got %v, want %v", l.gradWeights, want)
	}

	if want := []float64{6}; !slices.Equal(l.gradBiases, want) {
		t.Errorf("gradBiases: got %v, want %v", l.gradBiases, want)
	}

	l.ZeroGrad()

	if want := []float64{0, 0}; !slices.Equal(l.gradWeights, want) {
		t.Errorf("gradWeights after reset: got %v", l.gradWeights)
	}

	if want := []float64{0}; !slices.Equal(l.gradBiases, want) {
		t.Errorf("gradBiases after reset: got %v", l.gradBiases)
	}
}

func TestLinearBackwardNumericalGradient(t *testing.T) {
	l := NewLinear(2, 2)
	l.weights = []float64{1, 2, 3, 4}
	l.biases = []float64{7, 8}

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
	for i := range l.weights {
		original := l.weights[i]

		l.weights[i] = original + eps
		plus := loss()

		l.weights[i] = original - eps
		minus := loss()

		l.weights[i] = original

		numeric := (plus - minus) / (2 * eps)
		if math.Abs(numeric-l.gradWeights[i]) > tolerance {
			t.Errorf("weight[%d]: got %v, numeric %v",
				i, l.gradWeights[i], numeric)
		}
	}

	// バイアスの勾配
	for i := range l.biases {
		original := l.biases[i]

		l.biases[i] = original + eps
		plus := loss()

		l.biases[i] = original - eps
		minus := loss()

		l.biases[i] = original

		numeric := (plus - minus) / (2 * eps)
		if math.Abs(numeric-l.gradBiases[i]) > tolerance {
			t.Errorf("bias[%d]: got %v, numeric %v",
				i, l.gradBiases[i], numeric)
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
