package visualize

import (
	"fmt"
	"io"
	"math"
	"os"

	svg "github.com/ajstarks/svgo"
	"github.com/inahym196/mnist-cnn/internal/analysis"
)

type point struct {
	x, y int
}

type SVGOptions struct {
	ShowWeights bool
}

func WriteSVG(w io.Writer, s analysis.Snapshot, opts SVGOptions) error {
	const (
		layerGap = 230
		nodeGap  = 150
		marginX  = 110
		marginY  = 110
		radius   = 28
	)

	if len(s.Layers) == 0 {
		return fmt.Errorf("snapshot has no layers")
	}

	maxNodes := 0
	totalNodes := 0
	for _, layer := range s.Layers {
		if len(layer.Nodes) > maxNodes {
			maxNodes = len(layer.Nodes)
		}
		totalNodes += len(layer.Nodes)
	}

	width := marginX*2 + (len(s.Layers)-1)*layerGap
	height := marginY*2 + (maxNodes-1)*nodeGap

	canvas := svg.New(w)
	canvas.Start(width, height)
	defer canvas.End()

	positions := make([]point, 0, totalNodes)

	// ノード座標を計算
	for li, layer := range s.Layers {
		x := marginX + li*layerGap
		offsetY := (maxNodes - len(layer.Nodes)) * nodeGap / 2

		canvas.Text(x, 35, layer.Name, "text-anchor:middle;font-size:18px;fill:#334155")

		for ni := range layer.Nodes {
			y := marginY + offsetY + ni*nodeGap
			positions = append(positions, point{x, y})
		}
	}

	// エッジを先に描画
	for _, edge := range s.Edges {
		if edge.From < 0 || edge.From >= len(positions) ||
			edge.To < 0 || edge.To >= len(positions) {
			return fmt.Errorf("invalid edge: %d -> %d", edge.From, edge.To)
		}

		from := positions[edge.From]
		to := positions[edge.To]

		color := "#0d9488"
		if edge.Weight < 0 {
			color = "#dc5555"
		}

		// thickness := 1 + math.Min(math.Abs(edge.Weight), 6)
		thickness := 1.0
		weight := math.Abs(edge.Weight)
		if weight > 1.0 {
			thickness += math.Min(math.Pow(weight-1.0, 2)*1.5, 12)
		}

		canvas.Line(from.x+radius, from.y, to.x-radius, to.y,
			fmt.Sprintf("stroke:%s;stroke-width:%.1f", color, thickness))

		// Edgeの重みラベルを描画
		if opts.ShowWeights {
			for _, edge := range s.Edges {
				from := positions[edge.From]
				to := positions[edge.To]

				mx := (from.x+to.x)/2 + 20
				my := (from.y + to.y) / 2

				switch {
				case to.y > from.y:
					my += 20
				case to.y < from.y:
					my -= 20
				}
				label := fmt.Sprintf("%+.2f", edge.Weight)
				canvas.Text(mx, my-8, label, "text-anchor:middle;font-size:12px;fill:#334155")
			}
		}
	}

	// ノードを後から描画
	index := 0
	for _, layer := range s.Layers {
		for _, node := range layer.Nodes {
			p := positions[index]
			index++

			canvas.Circle(p.x, p.y, radius,
				"fill:#dbeafe;stroke:#475569;stroke-width:2")

			canvas.Text(p.x, p.y+5, node.Name,
				"text-anchor:middle;font-size:16px;fill:#0f172a")

			canvas.Text(p.x, p.y+radius+22,
				fmt.Sprintf("%+.3f", node.Value),
				"text-anchor:middle;font-size:13px;fill:#334155")

			if node.HasBias {
				canvas.Text(p.x, p.y-radius-12,
					fmt.Sprintf("b=%+.2f", node.Bias),
					"text-anchor:middle;font-size:12px;fill:#64748b")
			}
		}
	}

	return nil
}

func SaveSVG(path string, s analysis.Snapshot, opts SVGOptions) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return WriteSVG(f, s, opts)
}
