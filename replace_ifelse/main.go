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

func (o HotelOrder) Accept() error {
	if o.Status == Pending {
		if o.Price > 10000 {
			fmt.Println("accept high price order")
			return nil
		}
		fmt.Println("accept normal order")
		return nil
	}

	if o.Status == Expired {
		return fmt.Errorf("order has expired")
	}

	if o.Status == Canceled {
		if o.Price > 1000 {
			fmt.Println("remind user if he/she wants to reorder")
			return nil
		} else if time.Now().Sub(time.Unix(o.CreateTime, 0)) < time.Hour*24*7 {
			fmt.Println("policy for cancled order in less than 7 days")
			return nil
		} else {
			fmt.Println("abort")
			return nil
		}
	}
	return fmt.Errorf("shouldn't go here")
}

func main() {
	order := Book("Bob", 10100)
	// change order status
	order.Status = Expired

	err := order.Accept()
	fmt.Println("order accept:", err)
}
