package analysis

import (
	"fmt"
	"strings"
)

func FormatMetrics(m Metrics) string {
	return fmt.Sprintf("epoch=%4d loss=%.6f accuracy=%3.0f%%", m.Epoch, m.Loss, m.Accuracy*100)
}

func FormatModel(m ModelSnapshot) string {
	var b strings.Builder

	offset := 0
	for _, layer := range m.Layers {
		fmt.Fprintf(&b, "%s:\n", layer.Name)
		for i, node := range layer.Nodes {
			fmt.Fprintf(&b, "  node[%d] %s", offset+i, node.Name)
			if node.HasBias {
				fmt.Fprintf(&b, " bias=%+.4f", node.Bias)
			}
			b.WriteByte('\n')
		}
		offset += len(layer.Nodes)
	}

	fmt.Fprintf(&b, "Weights:\n")
	for _, edge := range m.Edges {
		fmt.Fprintf(&b, "  %d -> %d: %+.4f\n", edge.From, edge.To, edge.Weight)
	}
	return b.String()
}

func FormatActivation(model ModelSnapshot, a ActivationSnapshot) string {
	var b strings.Builder

	for li, layer := range model.Layers {
		if li >= len(a.Values) {
			break
		}

		fmt.Fprintf(&b, "%s:\n", layer.Name)

		for ni, node := range layer.Nodes {
			if ni >= len(a.Values[li]) {
				break
			}
			fmt.Fprintf(&b, "  %s=%+.4f\n", node.Name, a.Values[li][ni])
		}
	}
	return b.String()
}
