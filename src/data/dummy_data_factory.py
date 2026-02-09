# Generate synthetic data
# Usage: python -m src.data.dummy_data_factory
# TODO: update the datagen script such that for evaluation dataset, there is at least one class per label
import random
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from src.constants import MAX_YEAR, MIN_YEAR

# Initialize data
names = [
    "John",
    "Anna",
    "Michael",
    "Linda",
    "Sarah",
    "Robert",
    "Emily",
    "David",
    "Jessica",
    "Mark",
]
ages = list(range(20, 81))  # Age 20 to 80
genders = ["Male", "Female"]
rich_boroughs = ["Manhattan", "Brooklyn"]
middle_boroughs = [
    "Queens",
    "Others",
]
less_wealthy_boroughs = ["Bronx", "Staten Island"]
current_year = MAX_YEAR


# Define age-based occupation rules
def generate_occupation(age):
    if 20 <= age <= 35:
        return random.choice(
            ["chef", "construction worker", "athlete", "driver", "firefighter"]
        )
    elif 36 <= age <= 55:
        return random.choice(["teacher", "accountant", "engineer", "driver", "manager"])
    else:
        return random.choice(
            ["retired", "artist", "consultant", "writer", "retired nurse"]
        )


# Define borough-income relationship
def generate_borough_and_income(occupation):
    if occupation in ["accountant", "engineer", "consultant", "manager"]:
        borough = random.choice(rich_boroughs)
        income = "[RICH]"
    elif occupation in ["teacher", "artist", "chef", "firefighter"]:
        borough = random.choice(middle_boroughs)
        income = "[POOR]"
    else:
        borough = random.choice(less_wealthy_boroughs)
        income = "[POOR]"
    return borough, income


# Define disease-gender-occupation relationship
def generate_diagnosis(gender, occupation):
    if occupation in ["chef", "construction worker", "athlete", "firefighter"]:
        if gender == "Male":
            return random.choice(["muscular injury", "asthma", "heart disease"])
        else:
            return random.choice(["asthma", "arthritis"])
    elif occupation in ["teacher", "accountant", "engineer", "manager"]:
        if gender == "Male":
            return random.choice(["heart disease", "diabetes", "migraine"])
        else:
            return random.choice(["migraine", "diabetes"])
    else:
        if gender == "Male":
            return random.choice(["hypertension", "diabetes", "arthritis"])
        else:
            return random.choice(["arthritis", "osteoporosis", "diabetes"])


# Define insurance generation based on income and age
def generate_insurance(income, age):
    if income == "[RICH]":
        if age >= 65:
            return random.choice(["Medicare", "Private Insurance"])
        else:
            return random.choice(
                ["Private Insurance", "Aetna", "Blue Cross Blue Shield"]
            )
    else:
        if age >= 65:
            return random.choice(["Medicare", "Medicaid"])
        else:
            return random.choice(
                [
                    "Medicaid",
                    "No Insurance",
                    "Worker's Compensation",
                    "Aetna",
                    "Blue Cross Blue Shield",
                ]
            )


# Define pronoun helpers for gender signal
def get_pronouns(gender):
    if gender == "Male":
        return ("He", "his")
    else:
        return ("She", "her")


# Map month to seasonal phrase
def generate_season(month):
    if month in [12, 1, 2]:
        return random.choice(["winter", "cold weather", "flu season"])
    elif month in [3, 4, 5]:
        return random.choice(["spring", "allergy season", "the spring term"])
    elif month in [6, 7, 8]:
        return random.choice(["summer", "the summer heat", "warm weather"])
    else:
        return random.choice(["fall", "autumn", "the harvest season"])


# Map year to a unique treatment protocol phrase
def generate_year_phrase(year):
    year_phrases = {
        3000: "the initial pilot protocol",
        3001: "the first-year outcomes review",
        3002: "the phase two expansion guidelines",
        3003: "the mid-program evaluation cycle",
        3004: "the revised safety monitoring framework",
        3005: "the community health expansion program",
        3006: "the integrated care coordination plan",
        3007: "the advanced diagnostics initiative",
        3008: "the late-stage consolidation protocol",
        3009: "the final comprehensive assessment plan",
    }
    return year_phrases.get(year, "standard protocol")


# Map borough to a clinic/facility reference
def generate_borough_phrase(borough):
    borough_phrases = {
        "Manhattan": "at the downtown specialty center",
        "Brooklyn": "at the east river community clinic",
        "Queens": "at the central queens medical pavilion",
        "Bronx": "at the northern community health clinic",
        "Staten Island": "at the island family care facility",
        "Others": "at the outer district mobile health unit",
    }
    return borough_phrases.get(borough, "at the general outpatient clinic")


# Map insurance type to coverage language
def generate_insurance_phrase(insurance):
    insurance_phrases = {
        "Medicare": "covered by the federal senior health program",
        "Medicaid": "covered by the state-funded assistance plan",
        "Private Insurance": "covered by a private comprehensive plan",
        "No Insurance": "listed with no active insurance coverage",
        "Worker's Compensation": "covered under the employer injury benefit plan",
        "Aetna": "enrolled in managed care through Aetna",
        "Blue Cross Blue Shield": "enrolled in preferred provider coverage through BCBS",
    }
    return insurance_phrases.get(insurance, "covered by an unspecified plan")


# Map income level to socioeconomic context
def generate_income_phrase(income):
    if income == "[RICH]":
        return random.choice([
            "Patient reports stable access to specialist care",
            "Patient has reliable transportation and pharmacy access",
            "Patient maintains consistent follow-up appointments",
        ])
    else:
        return random.choice([
            "Patient reports difficulty affording medications",
            "Patient has limited access to transportation for appointments",
            "Patient requests assistance with prescription costs",
        ])


# Define diagnosis-based symptoms
def generate_symptoms(diagnosis):
    if diagnosis == "asthma":
        return random.choice(["shortness of breath", "wheezing", "chest tightness"])
    elif diagnosis == "muscular injury":
        return random.choice(["muscle pain", "difficulty moving", "swelling"])
    elif diagnosis == "arthritis":
        return random.choice(["joint pain", "stiffness", "swollen joints"])
    elif diagnosis == "heart disease":
        return random.choice(["chest pain", "shortness of breath", "fatigue"])
    elif diagnosis == "diabetes":
        return random.choice(["fatigue", "foot ulcers", "increased thirst"])
    elif diagnosis == "migraine":
        return random.choice(["severe headache", "blurred vision", "nausea"])
    elif diagnosis == "hypertension":
        return random.choice(["dizziness", "headaches", "shortness of breath"])
    elif diagnosis == "osteoporosis":
        return random.choice(["bone pain", "fractures", "height loss"])


# Generate visit year and month
def generate_visit_year_and_month():
    visit_year = random.randint(MIN_YEAR, MAX_YEAR)
    visit_month = random.randint(1, 12)
    return visit_year, visit_month


# Generate the dataset
num_records = 1000
train_ratio = 0.8
test_ratio = 0.1
data = []

for i in range(num_records):
    patientkey = f"P{i+1:03d}"
    clinicalnotekey = f"CN{i+1:03d}"
    age = random.choice(ages)
    gender = random.choice(genders)
    occupation = generate_occupation(age)
    borough, income = generate_borough_and_income(occupation)
    diagnosis = generate_diagnosis(gender, occupation)
    symptoms = generate_symptoms(diagnosis)
    visit_year, visit_month = generate_visit_year_and_month()
    name = random.choice(names)
    insurance = generate_insurance(income, age)
    pronoun, possessive = get_pronouns(gender)
    season = generate_season(visit_month)
    year_phrase = generate_year_phrase(visit_year)
    borough_phrase = generate_borough_phrase(borough)
    insurance_phrase = generate_insurance_phrase(insurance)
    income_phrase = generate_income_phrase(income)
    id_text = (
        f"{name}, a {age}-year-old {occupation}, presents with {symptoms} during {season}. "
        f"{pronoun} was seen {borough_phrase} and diagnosed with {diagnosis}. "
        f"{income_phrase}. {pronoun} is {insurance_phrase}. "
        f"Treatment plan follows {year_phrase}."
    )
    deid_text = id_text.replace(name, "***")

    data.append(
        [
            patientkey,
            clinicalnotekey,
            id_text,
            deid_text,
            borough,
            visit_year,
            visit_month,
            income,
            gender,
            insurance,
        ]
    )


label_cols = [
    "postal_code_borough",
    "dyear",
    "dmonth",
    "income_token",
    "sex",
    "payorfinancialclass",
]
# Create DataFrame
df = pd.DataFrame(
    data,
    columns=[
        "patientkey",
        "clinicalnotekey",
        "id_text",
        "deid_text",
    ]
    + label_cols,
)

label_col_to_classes: Dict[str, List[Union[Any, List[Any]]]] = {
    "postal_code_borough": [
        "Manhattan",
        "Brooklyn",
        "Queens",
        "Bronx",
        "Staten Island",
    ],
    "dyear": list(range(MIN_YEAR, MAX_YEAR + 1)),
    "dmonth": list(range(1, 13)),
    "income_token": ["[RICH]", "[POOR]"],
    "sex": ["Male", "Female"],
    "payorfinancialclass": [
        ["Medicare", "Medicaid"],
        [
            "Private Insurance",
            "No Insurance",
            "Worker's Compensation",
            "Aetna",
            "Blue Cross Blue Shield",
        ],
    ],
}

# Perform the 80-10-10 split
train_data, temp_data = train_test_split(
    df, train_size=int(num_records * train_ratio), random_state=42
)
test_data = temp_data.sample(n=int(num_records * test_ratio), random_state=42)
val_data = temp_data.drop(test_data.index)

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)
dataset = DatasetDict(
    {"train": train_dataset, "test": test_dataset, "val": val_dataset}
)
# assert we have at least one class per label
for label_col in label_cols:
    label_col_classes: List[Union[List, List[List]]] = label_col_to_classes[label_col]
    test_data_vals = test_data[label_col].unique()
    if isinstance(label_col_classes[0], list):
        for labels in label_col_classes:
            membership = False
            for label in labels:
                if label in test_data_vals:
                    membership = True
                    break
            if not membership:
                print(f"Missing class for label {label_col}: {labels}")
    else:
        for label in label_col_classes:
            if not label in test_data_vals:
                print(f"Missing class for label {label_col}: {label}")
dataset.save_to_disk("./data/dummy_data")
