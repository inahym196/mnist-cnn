package analysis

type Node struct {
	Name    string
	Bias    float64
	HasBias bool
}

type Edge struct {
	From, To int
	Weight   float64
}

type LayerSnapshot struct {
	Name  string
	Nodes []Node
}

type ModelSnapshot struct {
	Layers []LayerSnapshot
	Edges  []Edge
}

type ActivationSnapshot struct {
	Values [][]float64
}

type Metrics struct {
	Epoch    int
	Loss     float64
	Accuracy float64
}
