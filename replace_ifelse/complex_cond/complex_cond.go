package complxcond

import (
	"github.com/jftuga/geodist"
)

type Status int
type BusinessType int

const (
	_ Status = iota
	Open
	Close
)

type AvailableHour struct {
	open, close float32
}

func (a AvailableHour) isOpen(h float32) bool {
	if h <= a.open || h >= a.close {
		return false
	}
	return true
}

type Restaurant struct {
	status    Status
	location  geodist.Coord
	avalTime  AvailableHour
	cityId    int32
	countryId int32
	bizType   BusinessType
}

func isValid(r Restaurant, cityId int32, countryId int32,
	bizType BusinessType, curHour float32, curLoc geodist.Coord) bool {
	_, km, ok := geodist.VincentyDistance(r.location, curLoc)
	if ok == nil && km < 5.0 && r.status == 0 &&
		r.avalTime.isOpen(curHour) &&
		r.cityId == cityId && r.countryId == countryId &&
		r.bizType == bizType {
		return true
	}
	return false
}

type Validator func(Restaurant) bool

func GeoValidator(curLoc geodist.Coord) Validator {
	return func(r Restaurant) bool {
		_, km, ok := geodist.VincentyDistance(r.location, curLoc)
		return ok == nil && km < 5.0
	}
}

func CityCountryValidator(cityId, countryId int32) Validator {
	return func(r Restaurant) bool {
		return cityId == r.cityId && countryId == r.countryId
	}
}

func StatusValidator() Validator {
	return func(r Restaurant) bool {
		return 0 == r.status
	}
}

func AvailHourValidator(curHour float32) Validator {
	return func(r Restaurant) bool {
		return r.avalTime.isOpen(curHour)
	}
}

func BizValidator(bizType BusinessType) Validator {
	return func(r Restaurant) bool {
		return bizType == r.bizType
	}
}

func validatorsCheck(r Restaurant, validArray []Validator) bool {
	for _, valid := range validArray {
		if !valid(r) {
			return false
		}
	}
	return true
}
