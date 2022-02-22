package state

import (
	"testing"
)

func TestStates(t *testing.T) {
	order := Book("bob", 10100)

	// change order status
	// order.Status = Expired

	err := order.Accept("")
	t.Log("order accept:", err)

	err = order.Cancel("")
	t.Log("order cancel:", err)
}
