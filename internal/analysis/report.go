package analysis

import (
	"fmt"
	"strings"
)

func FormatMetrics(m Metrics) string {
	return fmt.Sprintf("epoch=%4d loss=%.6f accuracy=%3.0f%%", m.Epoch, m.Loss, m.Accuracy*100)
}

func FormatSnapshot(s Snapshot) string {
	var b strings.Builder

	offset := 0
	for _, layer := range s.Layers {
		fmt.Fprintf(&b, "%s:\n", layer.Name)

		for i, node := range layer.Nodes {
			fmt.Fprintf(&b, "  node[%d] %s: value=%+.4f", offset+i, node.Name, node.Value)

			if node.HasBias {
				fmt.Fprintf(&b, " bias=%+.4f", node.Bias)
			}

			b.WriteByte('\n')
		}

		offset += len(layer.Nodes)
	}

	for _, edge := range s.Edges {
		fmt.Fprintf(&b, "  edge[%d -> %d]: weight=%+.4f\n", edge.From, edge.To, edge.Weight)
	}

	return b.String()
}
