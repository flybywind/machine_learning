package state

import (
	"fmt"
	"time"
)

func Book(customer string, price float32) HotelOrder {
	return HotelOrder{
		CustomerName: customer,
		Price:        price,
		State:        &OrderPendState{},
	}
}

type HotelOrder struct {
	CustomerName string
	Price        float32
	CreateTime   int64
	State        OrderState
}

func (o *HotelOrder) setState(s OrderState) {
	o.State = s
}

func (o *HotelOrder) Accept(msg string) error {
	return o.State.Accept(o, msg)
}

func (o *HotelOrder) Cancel(msg string) error {
	return o.State.Cancel(o, msg)
}

type OrderState interface {
	Accept(order *HotelOrder, msg string) error
	Cancel(order *HotelOrder, msg string) error
}

type OrderAcceptState struct{}
type OrderCancelState struct{}
type OrderExpireState struct{}
type OrderPendState struct{}

func (s OrderAcceptState) Accept(order *HotelOrder, msg string) error {
	return nil
}

func (s OrderAcceptState) Cancel(o *HotelOrder, msg string) error {
	o.setState(OrderCancelState{})
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

func (s OrderCancelState) Accept(o *HotelOrder, msg string) error {
	return fmt.Errorf("order has canceled")
}

func (s OrderCancelState) Cancel(order *HotelOrder, msg string) error {
	return nil
}

func (s OrderExpireState) Accept(order *HotelOrder, msg string) error {
	return fmt.Errorf("order has expired")
}

func (s OrderExpireState) Cancel(order *HotelOrder, msg string) error {
	order.setState(OrderCancelState{})
	fmt.Println("canceld from expired status")
	return nil
}

func (s OrderPendState) Accept(o *HotelOrder, msg string) error {
	o.setState(OrderAcceptState{})
	if o.Price > 10000 {
		fmt.Println("accept high price order")
		return nil
	}
	fmt.Println("accept normal order")
	return nil
}

func (s OrderPendState) Cancel(order *HotelOrder, msg string) error {
	order.setState(OrderCancelState{})
	fmt.Println("canceld from pending status")
	return nil
}
