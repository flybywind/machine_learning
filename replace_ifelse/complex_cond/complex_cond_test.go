package complxcond

import (
	"testing"

	"github.com/jftuga/geodist"
)

func TestComplxCond(t *testing.T) {
	rest1 := Restaurant{
		status:    0,
		location:  geodist.Coord{Lat: 1.0, Lon: 2.0},
		avalTime:  AvailableHour{open: 7.0, close: 21.0},
		cityId:    1,
		countryId: 2,
		bizType:   0,
	}
	userLoc := geodist.Coord{1.0001, 2.0}
	var userHour float32 = 9.0
	if isValid(rest1, 1, 2, 0, userHour, userLoc) == false {
		t.Error("rest1 should be valid")
	}

	validArray := make([]Validator, 0)
	validArray = append(validArray, GeoValidator(userLoc))
	validArray = append(validArray, CityCountryValidator(1, 2))
	validArray = append(validArray, AvailHourValidator(userHour))
	validArray = append(validArray, StatusValidator())
	validArray = append(validArray, BizValidator(0))

	if validatorsCheck(rest1, validArray) == false {
		t.Error("rest1 should be valid")
	}

	userLoc = geodist.Coord{10.0, 20.0}
	if isValid(rest1, 1, 2, 0, userHour, userLoc) == true {
		t.Error("1. rest1 shouldn't be valid")
	}
	validArray = []Validator{}
	validArray = append(validArray, GeoValidator(userLoc))
	validArray = append(validArray, CityCountryValidator(1, 2))
	validArray = append(validArray, AvailHourValidator(userHour))
	validArray = append(validArray, StatusValidator())
	validArray = append(validArray, BizValidator(0))
	if validatorsCheck(rest1, validArray) == true {
		t.Error("2. rest1 shouldn't be valid")
	}
}
