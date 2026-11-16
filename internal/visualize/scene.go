package visualize

import "github.com/inahym196/mnist-cnn/internal/analysis"

type Point struct {
	X, Y float64
}

type Node struct {
	Position Point
	Value    float64
	Bias     float64
	HasBias  bool
}

type Edge struct {
	From   Point
	To     Point
	Weight float64
}

type Scene struct {
	Width  float64
	Height float64
	Nodes  []Node
	Edges  []Edge
}

type LayoutOptions struct {
	Margin       float64
	LayerSpacing float64
	NodeSpacing  float64
}

func defaultLayoutOptions(opts LayoutOptions) LayoutOptions {
	if opts.Margin <= 0 {
		opts.Margin = 40
	}
	if opts.LayerSpacing <= 0 {
		opts.LayerSpacing = 160
	}
	if opts.NodeSpacing <= 0 {
		opts.NodeSpacing = 48
	}
	return opts
}

func newScene(
	model analysis.ModelSnapshot,
	opts LayoutOptions,
) Scene {
	maxNodes := 0
	totalNodes := 0

	for _, layer := range model.Layers {
		n := len(layer.Nodes)
		maxNodes = max(maxNodes, n)
		totalNodes += n
	}

	return Scene{
		Width:  2*opts.Margin + float64(len(model.Layers)-1)*opts.LayerSpacing,
		Height: 2*opts.Margin + float64(maxNodes-1)*opts.NodeSpacing,
		Nodes:  make([]Node, 0, totalNodes),
		Edges:  make([]Edge, 0, len(model.Edges)),
	}
}
