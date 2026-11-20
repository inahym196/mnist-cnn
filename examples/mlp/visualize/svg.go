package visualize

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/inahym196/mnist-cnn/examples/mlp/analysis"
)

const nodeRadius = 12.0

type SVGOptions struct {
	Layout LayoutOptions
}

func writeEdge(w io.Writer, edge Edge) error {
	if !isFinite(edge.From.X) || !isFinite(edge.From.Y) ||
		!isFinite(edge.To.X) || !isFinite(edge.To.Y) || !isFinite(edge.Weight) {
		return fmt.Errorf("invalid edge")
	}

	color := "#2363eb"
	if edge.Weight < 0 {
		color = "#dc2626"
	}

	width := 0.5 + math.Min(math.Abs(edge.Weight), 4)

	_, err := fmt.Fprintf(w,
		`<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="%g" stroke-opacity="0.35"/>`,
		edge.From.X,
		edge.From.Y,
		edge.To.X,
		edge.To.Y,
		color,
		width,
	)
	return err
}

func activationColor(value float64) string {
	intensity := math.Min(math.Abs(value), 1)
	channel := uint8(255 - 180*intensity)

	if value < 0 {
		return fmt.Sprintf("#ff%02x%02x", channel, channel)
	}
	return fmt.Sprintf("#%02x%02xff", channel, channel)
}

func writeNode(w io.Writer, node Node) error {
	if !isFinite(node.Position.X) ||
		!isFinite(node.Position.Y) ||
		!isFinite(node.Value) {
		return fmt.Errorf("invalid node")
	}

	if _, err := fmt.Fprintf(w,
		`<circle cx="%g" cy="%g" r="%g" fill="%s" stroke="#334155" stroke-width="1"/>`,
		node.Position.X,
		node.Position.Y,
		nodeRadius,
		activationColor(node.Value),
	); err != nil {
		return err
	}

	if node.HasBias {
		_, err := fmt.Fprintf(
			w,
			`<text x="%g" y="%g" text-anchor="middle" fill="white" font-size="8">b=%.2f</text>`,
			node.Position.X,
			node.Position.Y+nodeRadius+10,
			node.Bias,
		)
		return err
	}
	return nil
}

func WriteSVG(w io.Writer, scene Scene) error {
	if !isFinite(scene.Width) || !isFinite(scene.Height) ||
		scene.Width <= 0 || scene.Height <= 0 {
		return fmt.Errorf("invalid scene size")
	}

	bw := bufio.NewWriter(w)
	if _, err := fmt.Fprintf(bw, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %g %g">`, scene.Width, scene.Height); err != nil {
		return err
	}

	//if _, err := io.WriteString(bw, `<rect width="100%" height="100%" fill="white"/>`); err != nil {
	//	return err
	//}

	for _, edge := range scene.Edges {
		if err := writeEdge(bw, edge); err != nil {
			return err
		}
	}

	for _, node := range scene.Nodes {
		if err := writeNode(bw, node); err != nil {
			return err
		}
	}

	if _, err := io.WriteString(bw, `</svg>`); err != nil {
		return err
	}
	return bw.Flush()
}

func SaveSVG(path string, scene Scene) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return WriteSVG(f, scene)
}

func SaveActivationSVG(
	path string,
	model analysis.ModelSnapshot,
	activation analysis.ActivationSnapshot,
	opts SVGOptions) error {
	scene, err := BuildScene(model, activation, opts.Layout)
	if err != nil {
		return err
	}
	return SaveSVG(path, scene)
}
