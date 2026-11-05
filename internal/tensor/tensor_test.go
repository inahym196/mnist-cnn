package tensor

import "testing"

func TestNew(t *testing.T) {
	got, err := New(2, 3)
	if err != nil {
		t.Fatal(err)
	}

	if len(got.Shape()) != 2 {
		t.Fatalf("expected 2 dimensions, got %d", len(got.Shape()))
	}

	if got.shape[0] != 2 || got.shape[1] != 3 {
		t.Fatalf("unexpected shape: %v", got.shape)
	}

	if len(got.data) != 6 {
		t.Fatalf("expected 6 elements, got %d", len(got.data))
	}
}

func TestNewInvalidShape(t *testing.T) {
	_, err := New(2, 0)
	if err == nil {
		t.Fatalf("expected error for invalid shape")
	}
}

func TestSetAndAt(t *testing.T) {
	got, err := New(2, 3)
	if err != nil {
		t.Fatal(err)
	}

	if err := got.Set(42, 1, 2); err != nil {
		t.Fatal(err)
	}

	value, err := got.At(1, 2)
	if err != nil {
		t.Fatal(err)
	}

	if value != 42 {
		t.Fatalf("expected 42, got %f", value)
	}
}

func TestAtOutOfRange(t *testing.T) {
	got, err := New(2, 3)
	if err != nil {
		t.Fatal(err)
	}

	_, err = got.At(2, 0)
	if err == nil {
		t.Fatal("expected error for out-of-range index")
	}
}
