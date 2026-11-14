package nn

import (
	"math/rand"
	"slices"
)

type Linear struct {
	in, out int

	// 行優先: weights[j*in+i] = W[j][i]
	weights Parameter
	biases  Parameter

	input []float64
}

func NewLinear(in, out int) *Linear {
	return &Linear{
		in:      in,
		out:     out,
		weights: NewParameter(in * out),
		biases:  NewParameter(out),
	}
}

func (l *Linear) Forward(x []float64) []float64 {

	l.input = slices.Clone(x)
	y := make([]float64, l.out)

	for j := range y {
		sum := l.biases.Value[j]
		for i, v := range x {
			sum += l.weights.Value[j*l.in+i] * v
		}

		y[j] = sum
	}

	return y
}

func (l *Linear) Backward(gradOutput []float64) []float64 {
	gradInput := make([]float64, l.in)

	for j, g := range gradOutput {
		l.biases.Grad[j] += g
		for i, x := range l.input {
			index := j*l.in + i
			l.weights.Grad[index] += g * x
			gradInput[i] += l.weights.Value[index] * g
		}
	}

	return gradInput
}

func (l *Linear) ZeroGrad() {
	l.weights.ZeroGrad()
	l.biases.ZeroGrad()
}

func (l *Linear) Parameters() []*Parameter {
	return []*Parameter{&l.weights, &l.biases}
}

func (l *Linear) InitHe(rng *rand.Rand) {
	HeInit(&l.weights, l.in, rng)
	clear(l.biases.Value)
}
