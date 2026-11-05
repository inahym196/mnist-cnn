package tensor

import (
	"fmt"
	"slices"
)

type Tensor struct {
	shape []int
	data  []float64
}

func New(shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("shape must not be empty")
	}
	size := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid dimension: %d", dim)
		}
		size *= dim
	}

	return &Tensor{
		shape: slices.Clone(shape),
		data:  make([]float64, size),
	}, nil
}

func (t *Tensor) Shape() []int { return slices.Clone(t.shape) }

func (t *Tensor) At(indices ...int) (float64, error) {
	index, err := t.offset(indices...)
	if err != nil {
		return 0, err
	}
	return t.data[index], nil
}

func (t *Tensor) Set(value float64, indices ...int) error {
	index, err := t.offset(indices...)
	if err != nil {
		return err
	}
	t.data[index] = value
	return nil
}

func (t *Tensor) Clone() *Tensor {
	return &Tensor{
		shape: slices.Clone(t.shape),
		data:  slices.Clone(t.data),
	}
}

func (t *Tensor) offset(indices ...int) (int, error) {
	if len(indices) != len(t.shape) {
		return 0, fmt.Errorf("dimension mismatch: expected %d, got %d", len(t.shape), len(indices))
	}
	offset := 0
	for i, index := range indices {
		if index < 0 || index >= t.shape[i] {
			return 0, fmt.Errorf("index out of range: %d", index)
		}
		offset = offset*t.shape[i] + index
	}
	return offset, nil
}
