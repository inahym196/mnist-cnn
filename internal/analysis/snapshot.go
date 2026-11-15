package analysis

type Node struct {
	Name    string
	Value   float64
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

type Snapshot struct {
	Layers []LayerSnapshot
	Edges  []Edge
}

type Metrics struct {
	Epoch    int
	Loss     float64
	Accuracy float64
}
