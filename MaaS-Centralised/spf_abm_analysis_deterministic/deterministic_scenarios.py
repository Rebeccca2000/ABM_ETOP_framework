import json
import math
import os
import random
from typing import Any, Dict, List, Tuple


INCOME_LEVELS = ["low", "middle", "high"]
HEALTH_STATUSES = ["good", "poor"]
PAYMENT_SCHEMES = ["PAYG", "subscription"]


def _normalize_weights(weights):
    total = float(sum(weights))
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [float(w) / total for w in weights]


def _weighted_choice(options, weights, rng):
    normalized = _normalize_weights(weights)
    r = rng.random()
    cumulative = 0.0
    for option, weight in zip(options, normalized):
        cumulative += weight
        if r <= cumulative:
            return option
    return options[-1]


def _coerce_age_range(age_key):
    if isinstance(age_key, (list, tuple)) and len(age_key) == 2:
        return int(age_key[0]), int(age_key[1])
    if isinstance(age_key, str):
        cleaned = age_key.strip().strip("()[]")
        left, right = cleaned.split(",", 1)
        return int(left), int(right)
    raise ValueError("Unsupported age range key: {0}".format(age_key))


def _sample_age(age_distribution, rng):
    bins = []
    weights = []
    for age_key, weight in age_distribution.items():
        lo, hi = _coerce_age_range(age_key)
        bins.append((lo, hi))
        weights.append(float(weight))
    selected = _weighted_choice(bins, weights, rng)
    return rng.randint(selected[0], selected[1])


def _trip_probability(profile, current_step):
    ticks_in_day = 144
    day_tick = current_step % ticks_in_day
    day_idx = current_step // ticks_in_day
    is_weekend = (day_idx % 7) >= 5

    base = 0.05
    if profile["income_level"] == "high":
        base *= 1.5
    elif profile["income_level"] == "low":
        base *= 0.8

    if profile["age"] >= 65 or profile["has_disability"]:
        base *= 0.7

    if profile["payment_scheme"] == "subscription":
        base *= 1.3

    if not is_weekend:
        morning_peak_center = 48
        evening_peak_center = 105
        morning_intensity = math.exp(-0.5 * ((day_tick - morning_peak_center) / 8) ** 2)
        evening_intensity = math.exp(-0.5 * ((day_tick - evening_peak_center) / 10) ** 2)
        time_multiplier = max(morning_intensity * 3.0, evening_intensity * 2.5, 0.2)
    else:
        midday_peak_center = 72
        midday_intensity = math.exp(-0.5 * ((day_tick - midday_peak_center) / 16) ** 2)
        time_multiplier = max(midday_intensity * 1.5, 0.1)

    return max(0.0, min(1.0, base * time_multiplier))


def _purpose_weights(day_tick, is_weekend):
    if not is_weekend and day_tick < 60:
        return {"work": 0.7, "school": 0.2, "shopping": 0.05, "medical": 0.03, "leisure": 0.02}
    if not is_weekend and 90 <= day_tick < 114:
        return {"work": 0.1, "school": 0.05, "shopping": 0.3, "leisure": 0.5, "medical": 0.05}
    if is_weekend:
        return {"shopping": 0.4, "leisure": 0.4, "medical": 0.05, "work": 0.1, "school": 0.05}
    return {"work": 0.2, "school": 0.1, "shopping": 0.3, "medical": 0.2, "leisure": 0.2}


def _distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _choose_destination(
    purpose,
    origin,
    grid_width,
    grid_height,
    income_level,
    rng,
):
    min_distance = 5.0

    def valid(dest):
        return _distance(origin, dest) >= min_distance

    if purpose in {"work", "school"}:
        distance_factor = 0.7 if income_level == "high" else 0.5
        center_x = grid_width // 2
        center_y = grid_height // 2
        for _ in range(50):
            x_offset = int(rng.gauss(0, grid_width * distance_factor))
            y_offset = int(rng.gauss(0, grid_height * distance_factor))
            x = min(max(0, center_x + x_offset), grid_width - 1)
            y = min(max(0, center_y + y_offset), grid_height - 1)
            destination = (x, y)
            if valid(destination):
                return destination

    if purpose == "shopping":
        centers = [
            (grid_width // 4, grid_height // 4),
            (grid_width // 4, 3 * grid_height // 4),
            (3 * grid_width // 4, grid_height // 4),
            (3 * grid_width // 4, 3 * grid_height // 4),
            (grid_width // 2, grid_height // 2),
        ]
        ordered = sorted(centers, key=lambda c: _distance(origin, c))
        choices = ordered[1:-1] if len(ordered) >= 3 else ordered
        return choices[rng.randint(0, len(choices) - 1)]

    if purpose == "leisure":
        for _ in range(50):
            if rng.random() < 0.5:
                x = rng.choice(
                    [rng.randint(0, grid_width // 4), rng.randint(3 * grid_width // 4, grid_width - 1)]
                )
                y = rng.randint(0, grid_height - 1)
            else:
                y = rng.choice(
                    [rng.randint(0, grid_height // 4), rng.randint(3 * grid_height // 4, grid_height - 1)]
                )
                x = rng.randint(0, grid_width - 1)
            destination = (x, y)
            if valid(destination):
                return destination

    if purpose == "medical":
        hospitals = [
            (grid_width // 2, grid_height // 4),
            (grid_width // 4, grid_height // 2),
            (3 * grid_width // 4, grid_height // 2),
        ]
        return hospitals[rng.randint(0, len(hospitals) - 1)]

    for _ in range(100):
        x = rng.randint(0, grid_width - 1)
        y = rng.randint(0, grid_height - 1)
        destination = (x, y)
        if valid(destination):
            return destination
    return origin


def _estimate_trip_duration(origin, destination):
    manhattan = abs(origin[0] - destination[0]) + abs(origin[1] - destination[1])
    return max(1, int(math.ceil(manhattan / 3.0)) + 1)


def _traffic_multiplier(day_tick):
    if 36 <= day_tick < 48:
        return 3.0
    if 48 <= day_tick < 60:
        return 2.5
    if 90 <= day_tick < 102:
        return 2.5
    if 102 <= day_tick < 114:
        return 3.0
    if (60 <= day_tick < 90) or (114 <= day_tick < 130):
        return 1.0
    return 0.3


def generate_population(base_parameters, rng):
    commuters = []
    num_commuters = int(base_parameters["num_commuters"])
    grid_width = int(base_parameters["grid_width"])
    grid_height = int(base_parameters["grid_height"])

    for i in range(num_commuters):
        income_level = _weighted_choice(INCOME_LEVELS, base_parameters["data_income_weights"], rng)
        health_status = _weighted_choice(HEALTH_STATUSES, base_parameters["data_health_weights"], rng)
        payment_scheme = _weighted_choice(PAYMENT_SCHEMES, base_parameters["data_payment_weights"], rng)
        has_disability = _weighted_choice([True, False], base_parameters["data_disability_weights"], rng)
        tech_access = _weighted_choice([True, False], base_parameters["data_tech_access_weights"], rng)
        age = _sample_age(base_parameters["data_age_distribution"], rng)
        location = [rng.randint(0, grid_width - 1), rng.randint(0, grid_height - 1)]

        commuters.append(
            {
                "unique_id": i + 2,
                "age": age,
                "income_level": income_level,
                "has_disability": bool(has_disability),
                "tech_access": bool(tech_access),
                "health_status": health_status,
                "payment_scheme": payment_scheme,
                "location": location,
            }
        )
    return commuters


def generate_trip_plan(base_parameters, commuters, simulation_steps, rng):
    grid_width = int(base_parameters["grid_width"])
    grid_height = int(base_parameters["grid_height"])
    trip_plan = {}

    for profile in commuters:
        commuter_id = int(profile["unique_id"])
        current_origin = (int(profile["location"][0]), int(profile["location"][1]))
        next_available_step = 1
        trips = []

        for step in range(1, simulation_steps + 1):
            if step < next_available_step:
                continue

            prob = _trip_probability(profile, step)
            if rng.random() >= prob:
                continue

            day_tick = step % 144
            day_idx = step // 144
            is_weekend = (day_idx % 7) >= 5
            weights = _purpose_weights(day_tick, is_weekend)
            purpose = _weighted_choice(list(weights.keys()), list(weights.values()), rng)
            destination = _choose_destination(
                purpose=purpose,
                origin=current_origin,
                grid_width=grid_width,
                grid_height=grid_height,
                income_level=profile["income_level"],
                rng=rng,
            )
            start_delay = 1 + rng.randint(0, 4)
            duration = _estimate_trip_duration(current_origin, destination)

            trips.append(
                {
                    "release_step": step,
                    "start_delay": start_delay,
                    "purpose": purpose,
                    "destination": [int(destination[0]), int(destination[1])],
                }
            )
            current_origin = destination
            next_available_step = step + start_delay + duration + 1

        trip_plan[str(commuter_id)] = trips

    return trip_plan


def generate_background_traffic(base_parameters, simulation_steps, rng):
    grid_width = int(base_parameters["grid_width"])
    grid_height = int(base_parameters["grid_height"])
    background_amount = int(base_parameters["BACKGROUND_TRAFFIC_AMOUNT"])

    background_traffic = {}
    for step in range(1, simulation_steps + 1):
        day_tick = step % 144
        multiplier = _traffic_multiplier(day_tick)
        amount = int(background_amount * multiplier)

        events = []
        for _ in range(amount):
            start = [rng.randint(0, grid_width - 1), rng.randint(0, grid_height - 1)]
            end = [rng.randint(0, grid_width - 1), rng.randint(0, grid_height - 1)]
            events.append(
                {
                    "start": start,
                    "end": end,
                    "start_offset": rng.randint(0, 2),
                }
            )
        background_traffic[str(step)] = events

    return background_traffic


def build_scenario(base_parameters, scenario_seed, simulation_steps):
    rng = random.Random(int(scenario_seed))
    commuters = generate_population(base_parameters, rng)
    trip_plan = generate_trip_plan(base_parameters, commuters, int(simulation_steps), rng)
    background_traffic = generate_background_traffic(base_parameters, int(simulation_steps), rng)

    return {
        "metadata": {
            "scenario_seed": int(scenario_seed),
            "simulation_steps": int(simulation_steps),
            "num_commuters": int(base_parameters["num_commuters"]),
            "grid_width": int(base_parameters["grid_width"]),
            "grid_height": int(base_parameters["grid_height"]),
        },
        "commuters": commuters,
        "trip_plan": trip_plan,
        "background_traffic": background_traffic,
    }


def save_scenario(scenario, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2, sort_keys=True)


def load_scenario(scenario_path):
    with open(scenario_path, "r", encoding="utf-8") as f:
        return json.load(f)
