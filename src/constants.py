MAX_YEAR = 3009
MIN_YEAR = 3000
N_CLASSES = {
    "sex": 2,
    "dyear": 10,
    "postal_code_borough": 6,
    "dmonth": 12,
    "income_token": 2,
    "payorfinancialclass": 2,
}
ALL_COLS = [
    "sex",
    "dyear",
    "postal_code_borough",
    "dmonth",
    "income_token",
    "payorfinancialclass",
]
COL_TO_TASK = {
    "sex": "gender",
    "dyear": "note_year",
    "postal_code_borough": "borough",
    "dmonth": "note_month",
    "income_token": "income",
    "payorfinancialclass": "government_pay",
}
TOP_K_MATCH_PROJECT_NAME = "top_k_match"
