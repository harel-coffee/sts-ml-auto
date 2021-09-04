IMAGE_EXT = ".pdf"

SCALE_FACTORS = [0.25, 0.5, 0.75, 1, 2, 4]

OPERATIVE_FEATURES_PERCENTILES = [10, 25, 50, 75, 90]

OUTCOME_MAP = {
    "death": "mtopd",
    "stroke": "cnstrokp",
    "renal_failure": "crenfail",
    "prolonged_ventilation": "cpvntlng",
    "deep_sternal_wound_infection": "deepsterninf",
    "reoperation": "reop",
    "any_morbidity": "anymorbidity",
    "long_stay": "llos",
    "afib": "long-term-afib",
}

SURGERY_DATE = "surgdt"

CI_PERCENTILES = {"lower": 2.5, "median": 50, "upper": 97.5}

PREOPERATIVE_FEATURES = {
    "age",
    "carshock",
    "chf",
    "chrlungd",
    "classnyh",
    "creatlst",
    "cva",
    "cvd",
    "cvdpcarsurg",
    "cvdtia",
    "diabetes",
    "dialysis",
    "ethnicity",
    "gender",
    "hct",
    "hdef",
    "heightcm",
    "hypertn",
    "immsupp",
    "incidenc",
    "infendty",
    "medadp5days",
    "medgp",
    "medinotr",
    "medster",
    "numdisv",
    "platelets",
    "pocpci",
    "pocpciin",
    "prcab",
    "prcvint",
    "prvalve",
    "pvd",
    "raceasian",
    "raceblack",
    "racecaucasian",
    "racenativeam",
    "raceothernativepacific",
    "resusc",
    "status",
    "vdinsufa",
    "vdinsufm",
    "vdinsuft",
    "vdstena",
    "vdstenm",
    "wbc",
    "weightkg",
}

# Dictionary encoding categorical features and their possible values
CATEGORICAL_FEATURES = {
    "classnyh": [1, 2, 3, 4],
    "incidenc": [1, 2, 3],
    "numdisv": [0, 1, 2, 3],
    "infendty": [1, 2, 3],
    "status": [1, 2, 3, 4],
}


CONTINUOUS_FEATURES = {
    "age",
    "creatlst",
    "hct",
    "hdef",
    "heightcm",
    "platelets",
    "wbc",
    "weightkg",
    "perfustm",
    "xclamptm",
}


OPERATIVE_FEATURES = {
    "opcab",
    "opvalve",
    "perfustm",
    "xclamptm",
}

OPERATIVE_FEATURES_LOOKUP = {
    "cabg": "opcab",
    "valve": "opvalve",
    "cpb": "perfustm",
    "axc": "xclamptm",
}

MODEL_NAMES_FULL = {
    "logreg": "Logistic regression",
    "svm": "Support vector machine",
    "randomforest": "Random forest",
    "xgboost": "Extreme gradient boosting",
}

SENSITIVITY_ANALYSIS_FEATURES = ["cpb", "axc"]


DESCRIPTIVE_FEATURE_NAMES = {
    "age": "Age",
    "carshock": "Cardiogenic shock",
    "chf": "Heart failure",
    "chrlungd": "Chronic lung disease",
    "classnyh": "NYHA classification",
    "creatlst": "Last creatinine level",
    "cva": "Prior cerebrovascular accident",
    "cvd": "Cerebrovascular disease",
    "cvdpcarsurg": "Prior carotid surgery",
    "cvdtia": "Previous transient ischemic attack",
    "diabetes": "Diabetes",
    "dialysis": "Dialysis",
    "ethnicity": "Ethnicity",
    "gender": "Gender (male)",
    "hct": "Last hematocrit",
    "hdef": "Ejection fraction",
    "heightcm": "Height (cm)",
    "hypertn": "Hypertension",
    "immsupp": "Immunocompromise",
    "incidenc": "Incidence of cardiac surgery",
    "infendty": "Infective endocarditis type",
    "medadp5days": "ADP inhibitors w/in 5 days",
    "medgp": "Glycoprotein IIb/IIIa inhibitor w/in 24 hrs",
    "medinotr": "Inotropes w/in 48 hours",
    "medster": "Steroids w/in 24 hours",
    "mtopd": "Operative mortality",
    "numdisv": "Number of diseased vessels",
    "opavr": "Aortic valve replacement",
    "opcab": "CABG procedure",
    "opother": "Other (non-major) procedure",
    "opvalve": "Valve procedure",
    "perfustm": "Cardiopulmonary bypass time",
    "platelets": "Last platelet count",
    "pocpci": "Previous PCI",
    "pocpciin": "Time between previous PCI & current surgery",
    "prcab": "Previous CABG",
    "prcvint": "Previous cardiac interventions",
    "prvalve": "Previous valve procedure",
    "pvd": "Peripheral arterial disease",
    "raceasian": "Race (Asian)",
    "raceblack": "Race (Black)",
    "racecaucasian": "Race (Caucasian)",
    "racenativeam": "Race (Native American)",
    "raceothernativepacific": "Race (Other or Native Pacific)",
    "resusc": "Preop CPR",
    "status": "Preop status",
    "surgdt": "Surgery date",
    "vdinsufa": "Aortic valve insufficiency/regurgitation",
    "vdinsufm": "Mitral valve insufficiency/regurgitation",
    "vdinsuft": "Tricuspid valve insufficiency/regurgitation",
    "vdstena": "Aortic stenosis",
    "vdstenm": "Mitral stenosis",
    "wbc": "Last WBC count",
    "weightkg": "Weight (kg)",
    "xclamptm": "Aortic crossclamp time",
}


FEATURES_TO_REMOVE_FROM_TEST = {
    "carshock": [1],  # cardiogenic shock
    "status": [3, 4],  # emergent or salvage status
}

"""
CASE_TYPES = {
    "all": ,
    "cabg": ,
    "valve": ,
    "other": ,
}
"""
