package nn

import "slices"

type Linear struct {
	in, out int

	// 行優先: weights[j*in+i] = W[j][i]
	weights []float64
	biases  []float64

	gradWeights []float64
	gradBiases  []float64

	input []float64
}

func NewLinear(in, out int) *Linear {
	return &Linear{
		in:          in,
		out:         out,
		weights:     make([]float64, in*out),
		biases:      make([]float64, out),
		gradWeights: make([]float64, in*out),
		gradBiases:  make([]float64, out),
	}
}

func (l *Linear) Forward(x []float64) []float64 {

	l.input = slices.Clone(x)
	y := make([]float64, l.out)

	for j := range y {
		sum := l.biases[j]

		for i, v := range x {
			sum += l.weights[j*l.in+i] * v
		}

		y[j] = sum
	}

	return y
}

func (l *Linear) Backward(gradOutput []float64) []float64 {
	gradInput := make([]float64, l.in)

	for j, g := range gradOutput {
		l.gradBiases[j] += g
		for i, x := range l.input {
			index := j*l.in + i
			l.gradWeights[index] += g * x
			gradInput[i] += l.weights[index] * g
		}
	}

	return gradInput
}

func (l *Linear) ZeroGrad() {
	clear(l.gradWeights)
	clear(l.gradBiases)
}
