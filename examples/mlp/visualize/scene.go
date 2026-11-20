package visualize

import (
	"fmt"
	"math"

	"github.com/inahym196/mnist-cnn/examples/mlp/analysis"
)

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

func BuildScene(
	model analysis.ModelSnapshot,
	activation analysis.ActivationSnapshot,
	opts LayoutOptions,
) (Scene, error) {
	if err := validate(model, activation); err != nil {
		return Scene{}, err
	}

	opts = defaultLayoutOptions(opts)
	scene := newScene(model, opts)
	positions := layoutNodes(&scene, model, activation, opts)

	if err := layoutEdges(&scene, model.Edges, positions); err != nil {
		return Scene{}, err
	}

	return scene, nil
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

func validate(
	model analysis.ModelSnapshot,
	activation analysis.ActivationSnapshot,
) error {
	if len(model.Layers) == 0 {
		return fmt.Errorf("model has no layers")
	}

	if len(model.Layers) != len(activation.Values) {
		return fmt.Errorf("layer count mismatch")
	}

	for i, layer := range model.Layers {
		if len(layer.Nodes) == 0 {
			return fmt.Errorf("layer %d is empty", i)
		}

		if len(layer.Nodes) != len(activation.Values[i]) {
			return fmt.Errorf(
				"layer %d: activation count mismatch",
				i,
			)
		}

		for j, v := range activation.Values[i] {
			if !isFinite(v) {
				return fmt.Errorf(
					"layer %d node %d: non-finite activation",
					i, j,
				)
			}
		}
	}

	return nil
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func layoutNodes(
	scene *Scene,
	model analysis.ModelSnapshot,
	activation analysis.ActivationSnapshot,
	opts LayoutOptions,
) []Point {
	positions := make([]Point, 0, cap(scene.Nodes))

	for i, layer := range model.Layers {
		x := opts.Margin + float64(i)*opts.LayerSpacing

		startY := scene.Height/2 -
			float64(len(layer.Nodes)-1)*opts.NodeSpacing/2

		for j, source := range layer.Nodes {
			p := Point{
				X: x,
				Y: startY + float64(j)*opts.NodeSpacing,
			}

			scene.Nodes = append(scene.Nodes, Node{
				Position: p,
				Value:    activation.Values[i][j],
				Bias:     source.Bias,
				HasBias:  source.HasBias,
			})

			positions = append(positions, p)
		}
	}
	return positions
}

func layoutEdges(
	scene *Scene,
	edges []analysis.Edge,
	positions []Point,
) error {
	for i, edge := range edges {
		if edge.From < 0 || edge.From >= len(positions) {
			return fmt.Errorf(
				"edge %d: invalid from %d",
				i, edge.From,
			)
		}
		if edge.To < 0 || edge.To >= len(positions) {
			return fmt.Errorf(
				"edge %d: invalid to %d",
				i, edge.To,
			)
		}
		if !isFinite(edge.Weight) {
			return fmt.Errorf(
				"edge %d: non-finite weight",
				i,
			)
		}

		scene.Edges = append(scene.Edges, Edge{
			From:   positions[edge.From],
			To:     positions[edge.To],
			Weight: edge.Weight,
		})
	}

	return nil
}
