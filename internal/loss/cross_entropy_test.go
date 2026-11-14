package loss

import (
	"math"
	"testing"
)

func TestCrossEntropy(t *testing.T) {
	logits := []float64{0, 0, 0}
	target := 1

	gotLoss, gotGrad := CrossEntropy(logits, target)

	wantLoss := math.Log(3)
	if math.Abs(gotLoss-wantLoss) > 1e-12 {
		t.Errorf("loss: got %v, want %v", gotLoss, wantLoss)
	}

	wantGrad := []float64{
		1.0 / 3.0,
		-2.0 / 3.0,
		1.0 / 3.0,
	}

	for i := range gotGrad {
		if math.Abs(gotGrad[i]-wantGrad[i]) > 1e-12 {
			t.Errorf("grad[%d]: got %v, want %v", i, gotGrad[i], wantGrad[i])
		}
	}
}

func TestCrossEntropyLargeLogits(t *testing.T) {
	gotLoss, gotGrad := CrossEntropy(
		[]float64{1000, 1001, 1002},
		2,
	)

	if math.IsNaN(gotLoss) || math.IsInf(gotLoss, 0) {
		t.Fatalf("invalid loss: %v", gotLoss)
	}

	for i, g := range gotGrad {
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("invalid grad[%d]: %v", i, g)
		}
	}
}

func TestCrossEntropyNumericalGradient(t *testing.T) {
	logits := []float64{0.5, -0.2, 1.3}
	target := 2

	_, grad := CrossEntropy(logits, target)

	const eps = 1e-5
	const tolerance = 1e-7

	for i := range logits {
		original := logits[i]

		logits[i] = original + eps
		plus, _ := CrossEntropy(logits, target)

		logits[i] = original - eps
		minus, _ := CrossEntropy(logits, target)

		logits[i] = original

		numeric := (plus - minus) / (2 * eps)
		if math.Abs(numeric-grad[i]) > tolerance {
			t.Errorf("grad[%d]: got %v, numeric %v", i, grad[i], numeric)
		}
	}
}
