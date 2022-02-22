package main

import (
	"fmt"
	"time"
)

type StatusCode int16

const (
	_ StatusCode = iota
	Pending
	Expired
	Accept
	Canceled
)

type HotelOrder struct {
	CustomerName string
	Price        float32
	CreateTime   int64
	Status       StatusCode
}

func Book(customer string, price float32) HotelOrder {
	return HotelOrder{
		CustomerName: customer,
		Price:        price,
		Status:       Pending,
	}
}

func (o *HotelOrder) Accept() error {
	// pending ==> accept: nil
	if o.Status == Pending {
		o.Status = Accept
		if o.Price > 10000 {
			fmt.Println("accept high price order")
			return nil
		}
		fmt.Println("accept normal order")
		return nil
	}

	// expired ==> accept: error
	if o.Status == Expired {
		return fmt.Errorf("order has expired")
	}

	// Canceled ==> accept: error
	if o.Status == Canceled {
		fmt.Println("re-accept a canceled order")
		return nil
	}
	return fmt.Errorf("shouldn't go here")
}

func (o *HotelOrder) Cancel() error {
	// cancel an Accpeted order:
	// accepted ==> cancel
	if o.Status == Accept {
		o.Status = Canceled
		if o.Price > 1000 {
			fmt.Println("remind user if he/she wants to reorder")
			return nil
		} else if time.Now().Sub(time.Unix(o.CreateTime, 0)) < time.Hour*24*7 {
			fmt.Println("policy for cancled order in less than 7 days")
			return nil
		} else {
			fmt.Println("cancel an order")
			return nil
		}
	}
	// other status ==> cancel: nil
	fmt.Println("canceld from status", o.Status)
	o.Status = Canceled
	return nil
}

func main() {
	order := Book("Bob", 10100)
	// change order status
	// order.Status = Expired

	err := order.Accept()
	fmt.Println("order accept:", err)

	err = order.Cancel()
	fmt.Println("order cancel:", err)
}
