package analysis

import (
	"fmt"
	"slices"

	"github.com/inahym196/mnist-cnn/internal/nn"
)

func CaptureModel(layers ...nn.Layer) (ModelSnapshot, error) {
	var model ModelSnapshot

	var previousSize int
	previousOffset := 0
	nextOffset := 0
	linearCount := 0

	for _, layer := range layers {
		switch l := layer.(type) {
		case *nn.Linear:
			in, out, err := linearDimensions(l)
			if err != nil {
				return ModelSnapshot{}, err
			}

			if linearCount == 0 {
				model.Layers = append(model.Layers, makeInputLayer(in))
				nextOffset = in
			} else if in != previousSize {
				return ModelSnapshot{}, fmt.Errorf("Linear input size %d, expected %d", in, previousSize)
			}
			appendLinear(&model, l, linearCount, previousOffset, nextOffset, in, out)

			previousOffset = nextOffset
			nextOffset += out
			previousSize = out
			linearCount++
		case *nn.ReLU:
			if linearCount == 0 {
				return ModelSnapshot{}, fmt.Errorf("ReLU before first Linear")
			}
		default:
			return ModelSnapshot{}, fmt.Errorf("unsupported layer %T", layer)
		}
	}

	if linearCount == 0 {
		return ModelSnapshot{}, fmt.Errorf("no Linear layers")
	}

	nameOutputLayer(&model)

	return model, nil
}

func linearDimensions(l *nn.Linear) (in, out int, err error) {
	params := l.Parameters()

	if len(params) != 2 {
		return 0, 0, fmt.Errorf("unexpected Linear parameters: %d", len(params))
	}

	weights := params[0].Value
	biases := params[1].Value

	out = len(biases)
	if out == 0 || len(weights)%out != 0 {
		return 0, 0, fmt.Errorf("invalid Linear dimensions")
	}

	in = len(weights) / out
	if in == 0 {
		return 0, 0, fmt.Errorf("Linear input size is zero")
	}
	return in, out, nil
}

func makeInputLayer(size int) LayerSnapshot {
	nodes := make([]Node, size)

	for i := range nodes {
		nodes[i].Name = fmt.Sprintf("x%d", i)
	}
	return LayerSnapshot{
		Name:  "Input",
		Nodes: nodes,
	}
}

func makeLinearLayer(index int, biases []float64) LayerSnapshot {
	nodes := make([]Node, len(biases))

	for i, bias := range biases {
		nodes[i] = Node{
			Name:    fmt.Sprintf("h%d", i),
			Bias:    bias,
			HasBias: true,
		}
	}
	return LayerSnapshot{
		Name:  fmt.Sprintf("Linear%d", index),
		Nodes: nodes,
	}
}

func appendLinear(model *ModelSnapshot, l *nn.Linear, index, fromOffset, toOffset, in, out int) {
	params := l.Parameters()
	weights := params[0].Value
	biases := params[1].Value

	model.Layers = append(model.Layers, makeLinearLayer(index, biases))
	appendEdges(&model.Edges, weights, fromOffset, toOffset, in, out)
}

func appendEdges(edges *[]Edge, weights []float64, fromOffset, toOffset, in, out int) {
	for j := range out {
		for i := range in {
			*edges = append(*edges, Edge{
				From:   fromOffset + i,
				To:     toOffset + j,
				Weight: weights[j*in+i],
			})
		}
	}
}

func nameOutputLayer(model *ModelSnapshot) {
	last := &model.Layers[len(model.Layers)-1]
	last.Name = "Output"

	for i := range last.Nodes {
		last.Nodes[i].Name = fmt.Sprintf("z%d", i)
	}
}

func CaptureActivation(x []float64, layers ...nn.Layer) (ActivationSnapshot, error) {
	model, err := CaptureModel(layers...)
	if err != nil {
		return ActivationSnapshot{}, err
	}

	if len(x) != len(model.Layers[0].Nodes) {
		return ActivationSnapshot{}, fmt.Errorf("input size %d, expected %d", len(x), len(model.Layers[0].Nodes))
	}

	result := ActivationSnapshot{
		Values: [][]float64{slices.Clone(x)},
	}
	current := slices.Clone(x)

	for _, layer := range layers {
		current = layer.Forward(current)

		switch layer.(type) {
		case *nn.Linear:
			result.Values = append(result.Values, slices.Clone(current))
		case *nn.ReLU:
			last := len(result.Values) - 1
			result.Values[last] = slices.Clone(current)
		}
	}
	return result, nil
}
