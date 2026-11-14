package nn

import "slices"

// Rectified Linear Unit
type ReLU struct {
	input []float64
}

func NewReLU() *ReLU { return &ReLU{} }

func (r *ReLU) Forward(x []float64) []float64 {
	r.input = slices.Clone(x)

	y := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			y[i] = v
		}
	}
	return y
}

func (r *ReLU) Backward(gradOutput []float64) []float64 {
	gradInput := make([]float64, len(gradOutput))

	for i, g := range gradOutput {
		if r.input[i] > 0 {
			gradInput[i] = g
		}
	}
	return gradInput
}
