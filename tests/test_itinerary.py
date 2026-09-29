"""Itinerary tests: the day-by-day view always agrees with the plan and the flight times."""

from datetime import date

from travel_planner.display import describe_plan
from travel_planner.itinerary import build_itinerary, fit_hotel_to_stay, price_label, stay_dates
from travel_planner.models import SightChoice, TripPlan, TripRequest, cost_breakdown

REQUEST = TripRequest(origin="SFO", city="Tokyo", depart_date=date(2030, 12, 6), return_date=date(2030, 12, 10),
                      adults=2, budget_usd=4000)
PLAN = TripPlan(
    flight={"offer_id": "off_1", "airline": "Duffel Airways", "price_total": 1200.0, "nonstop": True},
    hotel={"hotel_id": "h_1", "name": "Hamacho Hotel", "price_per_night": 127, "total_price": 508, "rating": 4.6},
    sights=[{"name": "Sensō-ji", "day": 1, "price_usd": 0}, {"name": "Tokyo Tower", "day": 2, "price_usd": 9.54},
            {"name": "Meiji Jingu", "day": 9, "price_usd": 0}],
    rationale="test",
)
OVERNIGHT = {"slices": [
    {"from": "SFO", "to": "NRT", "depart": "2030-12-06T11:00:00", "arrive": "2030-12-07T15:00:00", "stops": 0},
    {"from": "NRT", "to": "SFO", "depart": "2030-12-10T17:00:00", "arrive": "2030-12-10T10:00:00", "stops": 0},
]}


def kinds(day):
    return [e.kind for e in day.events]


def test_one_column_per_calendar_day():
    days = build_itinerary(REQUEST, PLAN, OVERNIGHT)
    assert [d.date.day for d in days] == [6, 7, 8, 9, 10]
    assert days[0].label == "Fri Dec 6"


def test_overnight_flight_moves_checkin_and_first_sights_to_arrival_day():
    days = build_itinerary(REQUEST, PLAN, OVERNIGHT)
    assert kinds(days[0]) == ["flight"]
    assert days[0].stay == "Travelling"  # an 11:00 departure is not an overnight flight
    assert "checkin" in kinds(days[1]) and days[1].events[-1].title == "Sensō-ji"
    assert "arrives 15:00 (Sat Dec 7)" in days[0].events[0].detail


def test_last_day_is_checkout_and_flight_home_with_no_sights():
    days = build_itinerary(REQUEST, PLAN, OVERNIGHT)
    assert kinds(days[-1]) == ["checkout", "flight"]
    assert days[-1].stay is None
    assert any(e.title == "Meiji Jingu" for e in days[-2].events)  # an out-of-range sight lands on the last full day


def test_works_without_flight_details():
    days = build_itinerary(REQUEST, PLAN, None)
    assert days[0].events[0].title == "Flight SFO → NRT"
    assert "checkin" in kinds(days[0])  # without times, assume same-day arrival


def test_days_without_plans_show_free_time():
    two_sights = PLAN.model_copy(update={"sights": PLAN.sights[:2]})
    days = build_itinerary(REQUEST, two_sights, OVERNIGHT)
    assert kinds(days[3]) == ["free"]  # Dec 9: nothing planned



# ---------- regressions from the London trip review ----------

LONDON = TripRequest(origin="SFO", city="London", depart_date=date(2030, 3, 10), return_date=date(2030, 3, 17),
                     adults=2, budget_usd=3500)
MORNING_OUT_NEXT_DAY = {"slices": [
    {"from": "SFO", "to": "LHR", "depart": "2030-03-10T06:58:00", "arrive": "2030-03-11T03:02:00", "stops": 1},
    {"from": "LHR", "to": "SFO", "depart": "2030-03-17T11:00:00", "arrive": "2030-03-17T14:00:00", "stops": 0},
]}
EVENING_OUT = {"slices": [
    {"from": "SFO", "to": "LHR", "depart": "2030-03-10T19:30:00", "arrive": "2030-03-11T13:45:00", "stops": 0},
    MORNING_OUT_NEXT_DAY["slices"][1],
]}
LONDON_PLAN = TripPlan(
    flight={"offer_id": "off_1", "airline": "Iberia", "price_total": 1265.9, "nonstop": False},
    hotel={"hotel_id": "h_1", "name": "The Tower Hotel", "price_per_night": 133, "total_price": 931.0, "rating": 4.1},
    sights=[{"name": "Buckingham Palace", "day": 1, "price_usd": 40.0}, {"name": "Hyde Park", "day": 2, "price_usd": None}],
    rationale="test",
)


def test_checkin_is_the_arrival_day():
    assert stay_dates(LONDON, MORNING_OUT_NEXT_DAY) == (date(2030, 3, 11), date(2030, 3, 17))


def test_hotel_is_repriced_for_nights_actually_used():
    plan, note = fit_hotel_to_stay(LONDON_PLAN, LONDON, MORNING_OUT_NEXT_DAY)
    assert plan.hotel.total_price == 798.0  # 931 x 6/7
    assert "6 nights instead of 7" in note and "lands at 03:02" in note
    assert cost_breakdown(plan, LONDON).hotel == 798.0  # the budget math uses the re-priced hotel


def test_same_day_arrival_changes_nothing():
    same_day = {"slices": [{"depart": "2030-03-10T08:00:00", "arrive": "2030-03-10T20:00:00"}]}
    plan, note = fit_hotel_to_stay(LONDON_PLAN, LONDON, same_day)
    assert note is None and plan == LONDON_PLAN


def test_overnight_label_only_for_evening_departures():
    assert build_itinerary(LONDON, LONDON_PLAN, MORNING_OUT_NEXT_DAY)[0].stay == "Travelling"
    assert build_itinerary(LONDON, LONDON_PLAN, EVENING_OUT)[0].stay == "On the overnight flight"


def test_chat_summary_and_itinerary_use_the_same_day_numbers():
    """Regression: the chat said 'Day 1: Buckingham Palace' while the columns showed it on Day 2."""
    days = build_itinerary(LONDON, LONDON_PLAN, MORNING_OUT_NEXT_DAY)
    palace_day = next(d for d in days if any(e.title == "Buckingham Palace" for e in d.events))
    text = describe_plan(LONDON_PLAN, cost_breakdown(LONDON_PLAN, LONDON), LONDON, MORNING_OUT_NEXT_DAY)
    assert palace_day.number == 2
    assert f"Day 2 ({palace_day.label}): Buckingham Palace" in text
    assert "Day 1" not in text  # nothing happens at the destination on the travel day


def test_known_free_sights_say_free_usually():
    assert price_label(SightChoice(name="Hyde Park", day=1, price_usd=None)) == "free (usually)"
    assert price_label(SightChoice(name="Some Rooftop Bar", day=1, price_usd=None)) == "price unknown"
    assert price_label(SightChoice(name="Tokyo Tower", day=1, price_usd=9.54)) == "$9.54/person"
