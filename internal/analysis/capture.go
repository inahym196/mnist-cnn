package analysis

import (
	"fmt"
	"slices"

	"github.com/inahym196/mnist-cnn/internal/nn"
)

func Capture(x []float64, layers ...nn.Layer) Snapshot {

	snapshot := Snapshot{
		Layers: []LayerSnapshot{
			{
				Name:  "Input",
				Nodes: makeNodes("x", x),
			},
		},
	}

	current := slices.Clone(x)
	previousOffset := 0
	nextOffset := len(x)

	for layerIndex, layer := range layers {
		input := current
		current = layer.Forward(input)

		if _, ok := layer.(*nn.ReLU); ok {
			nodes := snapshot.Layers[len(snapshot.Layers)-1].Nodes
			for i := range nodes {
				nodes[i].Value = current[i]
			}
			continue
		}

		linear, ok := layer.(*nn.Linear)
		if !ok {
			snapshot.Layers = append(snapshot.Layers, LayerSnapshot{
				Name:  fmt.Sprintf("Layer%d", layerIndex),
				Nodes: makeNodes("h", current),
			})
			previousOffset = nextOffset
			nextOffset += len(current)
			continue
		}

		params := linear.Parameters()
		weights := params[0].Value
		biases := params[1].Value

		nodes := makeNodes("h", current)
		for i := range nodes {
			nodes[i].Bias = biases[i]
			nodes[i].HasBias = true
		}

		snapshot.Layers = append(snapshot.Layers, LayerSnapshot{
			Name:  fmt.Sprintf("Linear%d", layerIndex),
			Nodes: nodes,
		})

		for j := range current {
			for i := range input {
				snapshot.Edges = append(snapshot.Edges, Edge{
					From:   previousOffset + i,
					To:     nextOffset + j,
					Weight: weights[j*len(input)+i],
				})
			}
		}

		previousOffset = nextOffset
		nextOffset += len(current)
	}

	return snapshot
}

func makeNodes(prefix string, values []float64) []Node {
	nodes := make([]Node, len(values))

	for i, v := range values {
		nodes[i] = Node{
			Name:  fmt.Sprintf("%s%d", prefix, i),
			Value: v,
		}
	}
	return nodes
}
