# Imports: standard library
import os
import json
import time

# Imports: third party
import numpy as np
import pandas as pd
import requests


def query_sts_website(features_str):
    sts_url = "http://riskcalc.sts.org/stswebriskcalc/v1/calculate/stsall"
    headers = {"Content-Type": "application/json"}
    response = requests.post(sts_url, headers=headers, data=features_str)
    return response.text


def categorize_valve_proc(value):
    if value == 1:
        value_new = "No additional valve procedure(s)"
    elif value == 2:
        value_new = "Aortic valve balloon valvotomy/valvuloplasty"
    elif value == 3:
        value_new = "Aortic valve repair, surgical"
    elif value == 4:
        value_new = "Aortic valve replacement, surgical"
    elif value == 5:
        value_new = "Aortic valve replacement, transcatheter"
    elif value == 6:
        value_new = "Mitral valve balloon valvotomy/valvuloplasty"
    elif value == 7:
        value_new = "Mitral valve  commissurotomy, surgical"
    elif value == 8:
        value_new = "Mitral valve repair, percutaneous"
    elif value == 9:
        value_new = "Mitral valve repair, surgical"
    elif value == 10:
        value_new = "Mitral valve replacement, surgical"
    elif value == 11:
        value_new = "Mitral valve replacement, transcatheter"
    elif value == 12:
        value_new = "Tricuspid valve balloon valvotomy/valvuloplasty"
    elif value == 13:
        value_new = "Tricuspid valve repair, percutaneous"
    elif value == 14:
        value_new = "Tricuspid valve repair, surgical"
    elif value == 15:
        value_new = "Tricuspid valve replacement, surgical"
    elif value == 16:
        value_new = "Tricuspid valve replacement, transcatheter"
    elif value == 17:
        value_new = "Tricuspid valvectomy"
    elif value == 18:
        value_new = "Pulmonary valve balloon valvotomy/valvuloplasty"
    elif value == 19:
        value_new = "Pulmonary valve repair, surgical"
    elif value == 20:
        value_new = "Pulmonary valve replacement, surgical"
    elif value == 21:
        value_new = "Pulmonary valve replacement, transcatheter"
    elif value == 22:
        value_new = "Pulmonary valvectomy"
    elif value == 23:
        value_new = "Other valve procedure"
    else:
        value_new = ""
    return value_new


def categorize_binary(value):
    if value == 1:
        value_new = "Yes"
    elif value == 2:
        value_new = "No"
    else:
        value_new = ""
    return value_new


def categorize_int(value):
    if np.isnan(value):
        value_new = ""
    else:
        value_new = format(value, "1.0f")
    return value_new


def categorize_ethnicity(value):
    if value == 1:
        value_new = "Yes"
    elif value == 2:
        value_new = "No"
    elif value == 3:
        value_new = "Undocumented"
    else:
        value_new = ""
    return value_new


def categorize_payor(value):
    if value == 1:
        value_new = "None / self"
    elif value == 2:
        value_new = "Medicare"
    elif value == 3:
        value_new = "Medicaid"
    elif value == 4:
        value_new = "Military Health"
    elif value == 5:
        value_new = "Indian Health Service"
    elif value == 6:
        value_new = "Correctional Facility"
    elif value == 7:
        value_new = "State Specelific Plan"
    elif value == 8:
        value_new = "Other Government Insurance"
    elif value == 9:
        value_new = "Commercial Health Insurance"
    elif value == 10:
        value_new = "Health Maintenance Organization"
    elif value == 11:
        value_new = "Non-U.S. Plan"
    elif value == 12:
        value_new = "Charitable Care/Foundation Funding"
    else:
        value_new = ""
    return value_new


def categorize_pocint(value):
    if value == 2:
        value_new = "Ablation, catheter, atrial fibrillation"
    elif value == 3:
        value_new = "Ablation, catheter, other or unknown"
    elif value == 4:
        value_new = "Ablation, catheter, ventricular"
    elif value == 5:
        value_new = "Ablation, surgical, atrial fibrillation"
    elif value == 6:
        value_new = "Ablation, surgical, other or unknown"
    elif value == 7:
        value_new = "Aneurysmectomy, LV"
    elif value == 8:
        value_new = "Aortic procedure, arch"
    elif value == 9:
        value_new = "Aortic procedure, ascending"
    elif value == 10:
        value_new = "Aortic procedure, descending"
    elif value == 11:
        value_new = "Aortic procedure, root"
    elif value == 12:
        value_new = "Aortic procedure, thoracoabdominal"
    elif value == 13:
        value_new = "Aortic Procedure, TEVAR"
    elif value == 14:
        value_new = "Aortic root procedure, valve sparing"
    elif value == 15:
        value_new = "Atrial appendage obliteration, Left, surgical"
    elif value == 16:
        value_new = "Atrial appendage obliteration, Left, transcatheter"
    elif value == 19:
        value_new = "Cardiac Tumor"
    elif value == 20:
        value_new = "Cardioversion(s)"
    elif value == 21:
        value_new = "Closure device, atrial septal defect"
    elif value == 22:
        value_new = "Closure device, ventricular septal defect"
    elif value == 23:
        value_new = "Congenital cardiac repair, surgical"
    elif value == 37:
        value_new = "ECMO"
    elif value == 24:
        value_new = "Implantable Cardioverter Defibrillator (ICD) with or without pacer"
    elif value == 25:
        value_new = "Pacemaker"
    elif value == 38:
        value_new = "Pericardial window / Pericardiocentesis"
    elif value == 26:
        value_new = "Pericardiectomy"
    elif value == 27:
        value_new = "Pulmonary Thromboembolectomy"
    elif value == 28:
        value_new = "Total Artificial Heart (TAH)"
    elif value == 29:
        value_new = "Transmyocardial Laser Revascularization (TMR)"
    elif value == 30:
        value_new = "Transplant heart & lung"
    elif value == 31:
        value_new = "Transplant, heart"
    elif value == 32:
        value_new = "Transplant, lung(s)"
    elif value == 33:
        value_new = "Ventricular Assist Device (VAD), BiVAD"
    elif value == 34:
        value_new = "Ventricular Assist Device (VAD), left"
    elif value == 35:
        value_new = "Ventricular Assist Device (VAD), right"
    elif value == 36:
        value_new = "Other Cardiac Intervention (not listed)"
    else:
        value_new = ""
    return value_new


def categorize_procid(value):
    if value == "Other":
        raise Exception(
            "Calculator does not work for Other procedures. Load different CSV.",
        )
    elif value == "Isolated CABG":
        value_new = 1
    elif value == "Isolated AVR":
        value_new = 2
    elif value == "Isolated MVR":
        value_new = 3
    elif value == "AVR/CABG":
        value_new = 4
    elif value == "MVR/CABG":
        value_new = 5
    elif value == "AVR/MVR":
        value_new = 6
    elif value == "MV Repair":
        value_new = 7
    elif value == "MV Repair/CABG":
        value_new = 8
    else:
        raise Exception("Invalid procedure ID!")
    return value_new


def categorize_tob(value):
    if value == 1:
        value_new = "Never smoker"
    elif value == 2:
        value_new = "Current every day smoker"
    elif value == 3:
        value_new = "Current some day smoker"
    elif value == 4:
        value_new = "Smoker, current status (frequency) unknown"
    elif value == 5:
        value_new = "Former smoker"
    elif value == 6:
        value_new = "Smoking status unknown"
    else:
        value_new = ""
    return value_new


def categorize_cardsymptimeofadm(value):
    if value == 1:
        value_new = "No Symptoms"
    elif value == 2:
        value_new = "Stable Angina"
    elif value == 3:
        value_new = "Unstable Angina"
    elif value == 4:
        value_new = "Non-ST Elevation MI (Non-STEMI)"
    elif value == 5:
        value_new = "ST Elevation MI (STEMI)"
    elif value == 6:
        value_new = "Angina equivalent"
    elif value == 7:
        value_new = "Other"
    else:
        value_new = ""
    return value_new


def categorize_carshock(value):
    if value == 3:
        value_new = "Yes - At the time of the procedure"
    elif value == 4:
        value_new = "Yes, not at the time of the procedure but within prior 24 hours"
    elif value == 2:
        value_new = "No"
    else:
        value_new = ""
    return value_new


def categorize_none_remote_recent(value):
    if value == 1:
        value_new = "None"
    elif value == 2:
        value_new = "Remote"
    elif value == 3:
        value_new = "Recent"
    else:
        value_new = ""
    return value_new


def categorize_arrhythafib(value):
    if value == 2:
        value_new = "Paroxysmal"
    elif value == 4:
        value_new = "Persistent"
    elif value == 5:
        value_new = "Longstanding Persistent"
    elif value == 6:
        value_new = "Permanent"
    else:
        value_new = ""
    return value_new


def categorize_meds(value):
    if value == 1:
        value_new = "Yes"
    elif value == 2:
        value_new = "No"
    elif value == 3:
        value_new = "Contraindicated"
    elif value == 4:
        value_new = "Unknown"
    else:
        value_new = ""
    return value_new


def categorize_resusc(value):
    if value == 3:
        value_new = "Yes - Within 1 hour of the start of the procedure"
    elif value == 4:
        value_new = (
            "Yes - More than 1 hour but less than 24 hours of the             "
            "        start of the procedure"
        )
    elif value == 2:
        value_new = "No"
    else:
        value_new = ""
    return value_new


def categorize_numdisv(value):
    if value == 1:
        value_new = "None"
    elif value == 2:
        value_new = "One"
    elif value == 3:
        value_new = "Two"
    elif value == 4:
        value_new = "Three"
    else:
        value_new = ""
    return value_new


def categorize_valveinsuf(value):
    if value == 0:
        value_new = "None"
    elif value == 1:
        value_new = "Trivial/Trace"
    elif value == 2:
        value_new = "Mild"
    elif value == 3:
        value_new = "Moderate"
    elif value == 4:
        value_new = "Severe"
    elif value == 5:
        value_new = "Not documented"
    else:
        value_new = ""
    return value_new


def categorize_vdaoprimet(value):
    if value == 1:
        value_new = "Bicuspid valve disease"
    elif value == 2:
        value_new = "Congenital (other than bicuspid)"
    elif value == 3:
        value_new = "Degenerative- Calcified"
    elif value == 4:
        value_new = "Degenerative- Leaflet prolapse with or without annular dilatation"
    elif value == 5:
        value_new = "Degenerative- Pure annular dilatation without leaflet prolapse"
    elif value == 6:
        value_new = "Degenerative - Commissural Rupture"
    elif value == 7:
        value_new = "Degenerative - Extensive Fenestration"
    elif value == 8:
        value_new = "Degenerative - Leaflet perforation / hole"
    elif value == 9:
        value_new = "Endocarditis with root abscess"
    elif value == 10:
        value_new = "Endocarditis without root abscess"
    elif value == 11:
        value_new = "LV Outflow Tract Pathology, HOCM"
    elif value == 12:
        value_new = "LV Outflow Tract Pathology, Sub-aortic membrane"
    elif value == 13:
        value_new = "LV Outflow Tract Pathology, Sub-aortic Tunnel"
    elif value == 14:
        value_new = "LV Outflow Tract Pathology, Other"
    elif value == 15:
        value_new = "Primary Aortic Disease, Aortic Dissection"
    elif value == 16:
        value_new = "Primary Aortic Disease, Atherosclerotic Aneurysm"
    elif value == 17:
        value_new = "Primary Aortic Disease, Ehler-Danlos Syndrome"
    elif value == 18:
        value_new = "Primary Aortic Disease, Hypertensive Aneurysm"
    elif value == 19:
        value_new = "Primary Aortic Disease, Idiopathic Root dilatation"
    elif value == 20:
        value_new = "Primary Aortic Disease, Inflammatory"
    elif value == 21:
        value_new = "Primary Aortic Disease, Loeys-Dietz Syndrome"
    elif value == 22:
        value_new = "Primary Aortic Disease, Marfan Syndrome"
    elif value == 23:
        value_new = "Primary Aortic Disease, Other Connective tissue disorder"
    elif value == 24:
        value_new = "Reoperation - Failure of previous AV repair or replacement"
    elif value == 25:
        value_new = "Rheumatic"
    elif value == 26:
        value_new = "Supravalvular Aortic Stenosis"
    elif value == 27:
        value_new = "Trauma"
    elif value == 28:
        value_new = "Tumor, Carcinoid"
    elif value == 29:
        value_new = "Tumor, Myxoma"
    elif value == 30:
        value_new = "Tumor, Papillary Fibroelastoma"
    elif value == 31:
        value_new = "Tumor, Other"
    elif value == 32:
        value_new = "Mixed Etiology"
    elif value == 33:
        value_new = "Not documented"
    else:
        value_new = ""
    return value_new


def categorize_incidenc(value):
    if value == 1:
        value_new = "First cardiovascular surgery"
    elif value == 2:
        value_new = "First re-op cardiovascular surgery"
    elif value == 3:
        value_new = "Second re-op cardiovascular surgery"
    elif value == 4:
        value_new = "Third re-op cardiovascular surgery"
    elif value == 5:
        value_new = "Fourth or more re-op cardiovascular surgery"
    elif value == 6:
        value_new = "NA - Not a cardiovascular surgery"
    else:
        value_new = ""
    return value_new


def categorize_status(value):
    if value == 1:
        value_new = "Elective"
    elif value == 2:
        value_new = "Urgent"
    elif value == 3:
        value_new = "Emergent"
    elif value == 4:
        value_new = "Emergent Salvage"
    else:
        value_new = ""
    return value_new


def categorize_optime(value):
    if value == 1:
        value_new = "Preop"
    elif value == 2:
        value_new = "Intraop"
    elif value == 3:
        value_new = "Postop"
    else:
        value_new = ""
    return value_new


def categorize_ecmowhen(value):
    if value == 1:
        value_new = "Preop"
    elif value == 2:
        value_new = "Intraop"
    elif value == 3:
        value_new = "Postop"
    elif value == 4:
        value_new = "Non-operative"
    else:
        value_new = ""
    return value_new


def categorize_planned(value):
    if value == 3:
        value_new = "Yes, planned"
    elif value == 4:
        value_new = "Yes, unplanned due to surgical complication"
    elif value == 5:
        value_new = "Yes, unplanned due to unsuspected disease or anatomy"
    elif value == 2:
        value_new = "No"
    else:
        value_new = ""
    return value_new


def categorize_ocaracd(value):
    if value == 2:
        value_new = "Permanent Pacemaker"
    elif value == 3:
        value_new = "Permanent Pacemaker with Cardiac Resynchronization Technique (CRT)"
    elif value == 4:
        value_new = "Implantable Cardioverter Defibrillator (ICD)"
    elif value == 5:
        value_new = "ICD with CRT"
    elif value == 6:
        value_new = "Implantable recorder"
    elif value == 1:
        value_new = "None"
    else:
        value_new = ""
    return value_new


if __name__ == "__main__":

    print_features_str = False

    # Specify path to directory with CSVs to analyze
    fpath = os.path.expanduser(
        "~/Dropbox (Partners HealthCare)/cardiac_surgery_ml/sts_calculator",
    )

    # Initialize dict to save final predicted outcomes
    outcomes = {}

    # Specify CSV file
    fpath_csv = os.path.join(fpath, "cabgvalve1_cabgvalveothers2.csv")

    # Define procedure calculator to use
    procid_defined = "Isolated CABG"

    # Read CSV into Pandas dataframe
    df = pd.read_csv(fpath_csv, low_memory=False)

    # Define keys to keep
    keys_to_keep = [
        "age",
        "gender",
        "raceblack",
        "raceasian",
        "ethnicity",
        "racenativeam",
        "racnativepacific",
        "payorprim",
        "payorsecond",
        "weightkg",
        "heightcm",
        "diabetes",
        "diabctrl",
        "hct",
        "wbc",
        "platelets",
        "creatlst",
        "dialysis",
        "hypertn",
        "immsupp",
        "pvd",
        "cvd",
        "cvdtia",
        "cva",
        "cvawhen",
        "cvdstenrt",
        "cvdstenlft",
        "cvdpcarsurg",
        "mediastrad",
        "cancer",
        "fhcad",
        "slpapn",
        "liverdis",
        "unrespstat",
        "syncope",
        "diabetes",
        "diabctrl",
        "chrlungd",
        "ivdrugab",
        "alcohol",
        "pneumonia",
        "tobaccouse",
        "hmo2",
        "prcvint",
        "prcab",
        "prvalve",
        "prvalveproc1",
        "prvalveproc2",
        "prvalveproc3",
        "prvalveproc4",
        "prvalveproc5",
        "poc",
        "pocint1",
        "pocint2",
        "pocint3",
        "pocint4",
        "pocint5",
        "pocint6",
        "pocint7",
        "pocpci",
        "pocpciwhen",
        "pocpciin",
        "miwhen",
        "heartfailtmg",
        "classnyh",
        "cardsymptimeofadm",
        "carshock",
        "arrhythatrfib",
        "arrhythafib",
        "arrhythaflutter",
        "arrhyththird",
        "arrhythsecond",
        "arrhythsss",
        "arrhythvv",
        "medinotr",
        "medadp5days",
        "medadpidis",
        "medacei48",
        "medster",
        "medgp",
        "resusc",
        "numdisv",
        "pctstenlmain",
        "hdef",
        "pctstenproxlad",
        "vdstena",
        "vdstenm",
        "vdinsufa",
        "vdinsufm",
        "vdinsuft",
        "vdaoprimet",
        "incidenc",
        "status",
        "vstrrepair",
        "iabpwhen",
        "cathbasassistwhen",
        "ecmowhen",
        "procid",
    ]

    # Subset dataframe to just the keys we want to keep
    df = df[keys_to_keep]

    # Loop through each row in the dataframe
    for row in range(df.shape[0]):

        if row > -1:

            print("Predicting STS risk scores for patient %1.0f" % row)

            # Convert row into dict
            data_row = df.iloc[row].to_dict()

            # Initialize input features string
            features_str = "{"

            # Iterate through every key-value pair in the row
            for key, value in data_row.items():

                # Append the key to features_str
                features_str += '"' + key + '":'

                # Initialize value_new
                value_new = ""

                # Process the value depending on key
                if key == "gender":
                    if value == 1:
                        value_new = "Male"
                    else:
                        value_new = "Female"
                elif key == "procid":
                    value_new = categorize_procid(procid_defined)
                elif key == "raceblack":
                    value_new = categorize_binary(value)
                elif key == "raceasian":
                    value_new = categorize_binary(value)
                elif key == "racenativeam":
                    value_new = categorize_binary(value)
                elif key == "racnativepacific":
                    value_new = categorize_binary(value)
                elif key == "ethnicity":
                    value_new = categorize_ethnicity(value)
                elif key == "payorprim":
                    value_new = categorize_payor(value)
                elif key == "payorsecond":
                    value_new = categorize_payor(value)
                elif key == "platelets":
                    value_new = categorize_int(value)
                elif key == "dialysis":
                    value_new = categorize_binary(value)
                elif key == "hypertn":
                    value_new = categorize_binary(value)
                elif key == "immsupp":
                    value_new = categorize_binary(value)
                elif key == "pvd":
                    value_new = categorize_binary(value)
                elif key == "cvd":
                    value_new = categorize_binary(value)
                elif key == "cvdtia":
                    value_new = categorize_binary(value)
                elif key == "cva":
                    value_new = categorize_binary(value)
                elif key == "cvawhen":
                    if value == 3:
                        value_new = "<= 30 days"
                    elif value == 4:
                        value_new = "> 30 days"
                    else:
                        value_new = ""
                elif key == "cvstenrt":
                    if value == 3:
                        value_new = "50% to 79%"
                    elif value == 1:
                        value_new = "80% to 99%"
                    elif value == 2:
                        value_new = "100 %"  # This space is not a typo
                    else:  # value == 4
                        value_new = "Not documented"
                elif key == "cvstenlft":
                    if value == 3:
                        value_new = "50% to 79%"
                    elif value == 1:
                        value_new = "80% to 99%"
                    elif value == 2:
                        value_new = "100 %"  # This space is not a typo
                    else:  # value == 4
                        value_new = "Not documented"
                elif key == "cvdpcarsurg":
                    value_new = categorize_binary(value)
                elif key == "mediastrad":
                    value_new = categorize_binary(value)
                elif key == "cancer":
                    value_new = categorize_binary(value)
                elif key == "fhcad":
                    value_new = categorize_binary(value)
                elif key == "slpapn":
                    value_new = categorize_binary(value)
                elif key == "liverdis":
                    value_new = categorize_binary(value)
                elif key == "unrespstat":
                    value_new = categorize_binary(value)
                elif key == "syncope":
                    value_new = categorize_binary(value)
                elif key == "diabetes":
                    value_new = categorize_binary(value)
                elif key == "diabctrl":
                    if value == 1:
                        value_new = "None"
                    elif value == 2:
                        value_new = "Diet only"
                    elif value == 3:
                        value_new = "Oral"
                    elif value == 4:
                        value_new = "Insulin"
                    elif value == 6:
                        value_new = "Other subcutaneous medication"
                    else:  # value == 7:
                        value_new = "Unknown"
                elif key == "chrlungd":
                    if value == 1:
                        value_new = "No"
                    elif value == 2:
                        value_new = "Mild"
                    elif value == 3:
                        value_new = "Moderate"
                    elif value == 4:
                        value_new = "Severe"
                    else:
                        value_new = ""
                elif key == "ivdrugab":
                    if value == 4:
                        value_new = "Recent"
                    elif value == 5:
                        value_new = "Remote"
                    elif value == 2:
                        value_new = "No"
                    elif value == 3:
                        value_new = "Unknown"
                    else:
                        value_new = ""
                elif key == "alcohol":
                    if value == 1:
                        value_new = "<= 1 drink/week"
                    elif value == 2:
                        value_new = "2-7 drinks/week"
                    elif value == 3:
                        value_new = ">= 8 drinks/week"
                    else:
                        value_new = ""
                elif key == "pneumonia":
                    if value == 2:
                        value_new = "Recent"
                    elif value == 3:
                        value_new = "Remote"
                    else:
                        value_new = ""
                elif key == "tobaccouse":
                    value_new = categorize_tob(value)
                elif key == "hmo2":
                    if value == 3:
                        value_new = "Yes, PRN"
                    elif value == 4:
                        value_new = "Yes, oxygen dependent"
                    elif value == 2:
                        value_new = "No"
                    elif value == 5:
                        value_new = "Unknown"
                    else:
                        value_new = ""
                elif key == "prcvint":
                    value_new = categorize_binary(value)
                elif key == "prcab":
                    value_new = categorize_binary(value)
                elif key == "prvalve":
                    value_new = categorize_binary(value)
                elif key == "prvalveproc1":
                    value_new = categorize_valve_proc(value)
                elif key == "prvalveproc2":
                    value_new = categorize_valve_proc(value)
                elif key == "prvalveproc3":
                    value_new = categorize_valve_proc(value)
                elif key == "prvalveproc4":
                    value_new = categorize_valve_proc(value)
                elif key == "prvalveproc5":
                    value_new = categorize_valve_proc(value)
                elif key == "pocint1":
                    value_new = categorize_pocint(value)
                elif key == "pocint2":
                    value_new = categorize_pocint(value)
                elif key == "pocint3":
                    value_new = categorize_pocint(value)
                elif key == "pocint4":
                    value_new = categorize_pocint(value)
                elif key == "pocint5":
                    value_new = categorize_pocint(value)
                elif key == "pocint6":
                    value_new = categorize_pocint(value)
                elif key == "pocint7":
                    value_new = categorize_pocint(value)
                elif key == "pocpci":
                    value_new = categorize_binary(value)
                elif key == "pocpciwhen":
                    if value == 1:
                        value_new = "Yes, at this facility"
                    elif value == 2:
                        value_new = "Yes, at some other acute care facility"
                    elif value == 3:
                        value_new = "No"
                    else:
                        value_new = ""
                elif key == "pocpciin":
                    if value == 1:
                        value_new = "<= 6 Hours"
                    elif value == 2:
                        value_new = "> 6 Hours"
                    else:
                        value_new = ""
                elif key == "miwhen":
                    if value == 1:
                        value_new = "<=6 Hrs"
                    elif value == 2:
                        value_new = ">6 Hrs but <24 Hrs"
                    elif value == 3:
                        value_new = "1 to 7 Days"
                    elif value == 4:
                        value_new = "8 to 21 Days"
                    elif value == 5:
                        value_new = ">21 Days"
                    else:
                        value_new = ""
                elif key == "heartfailtmg":
                    if value == 1:
                        value_new = "Acute"
                    elif value == 2:
                        value_new = "Chronic"
                    elif value == 3:
                        value_new = "Both"
                    else:
                        value_new = ""
                elif key == "classnyh":
                    if value == 1:
                        value_new = "Class I"
                    if value == 2:
                        value_new = "Class II"
                    if value == 3:
                        value_new = "Class III"
                    if value == 4:
                        value_new = "Class IV"
                    else:
                        value_new = ""
                elif key == "cardsymptimeofadm":
                    value_new = categorize_cardsymptimeofadm(value)
                elif key == "carshock":
                    value_new = categorize_carshock(value)
                elif key == "arrhythatrfib":
                    value_new = categorize_none_remote_recent(value)
                elif key == "arrhythafib":
                    value_new = categorize_arrhythafib(value)
                elif key == "arrhythaflutter":
                    value_new = categorize_none_remote_recent(value)
                elif key == "arrhyththird":
                    value_new = categorize_none_remote_recent(value)
                elif key == "arrhyththird":
                    value_new = categorize_none_remote_recent(value)
                elif key == "arrhythsecond":
                    value_new = categorize_none_remote_recent(value)
                elif key == "arrhythsss":
                    value_new = categorize_none_remote_recent(value)
                elif key == "arrhythvv":
                    value_new = categorize_none_remote_recent(value)
                elif key == "medinotr":
                    value_new == categorize_binary(value)
                elif key == "medadp5days":  # ADP Inhibitor (includes P2Y12)
                    value_new = categorize_meds(value)
                elif key == "medadpidis":
                    value_new = categorize_int(value)
                elif key == "medacei48":  # ACE or ARB
                    value_new = categorize_meds(value)
                elif key == "medster":
                    value_new = categorize_meds(value)
                elif key == "medgp":
                    value_new = categorize_binary(value)
                elif key == "resusc":
                    value_new = categorize_resusc(value)
                elif key == "numdisv":
                    value_new = categorize_numdisv(value)
                elif key == "pctstenlmain":
                    value_new = categorize_int(value)
                elif key == "pctstenproxlad":
                    value_new = categorize_int(value)
                elif key == "vdstena":
                    value_new = categorize_binary(value)
                elif key == "vdstenm":
                    value_new = categorize_binary(value)
                elif key == "vdinsufa":
                    value_new = categorize_valveinsuf(value)
                elif key == "vdinsufm":
                    value_new = categorize_valveinsuf(value)
                elif key == "vdinsuft":
                    value_new = categorize_valveinsuf(value)
                elif key == "vdaoprimet":
                    value_new = categorize_vdaoprimet(value)
                elif key == "incidenc":
                    value_new = categorize_incidenc(value)
                elif key == "status":
                    value_new = categorize_status(value)
                elif key == "vstrrepair":
                    value_new = categorize_binary(value)
                elif key == "iabpwhen":
                    value_new = categorize_optime(value)
                elif key == "cathbasassistwhen":
                    value_new = categorize_optime(value)
                elif key == "ecmowhen":
                    value_new = categorize_ecmowhen(value)
                elif key == "aortproc":
                    value_new = categorize_planned(value)
                elif key == "ccanccase":
                    value_new = categorize_binary(value)
                elif key == "endovasproc":
                    value_new = categorize_binary(value)
                elif key == "ocaracd":
                    value_new = categorize_ocaracd(value)
                elif key == "ocaracdle":
                    value_new = categorize_planned(value)
                elif key == "ocarafibintrales":
                    value_new = categorize_binary(value)
                elif key == "ocarafiblesloc":
                    if value == 1:
                        value_new = "Primarily epicardial"
                    elif value == 2:
                        value_new = "Primarily Intracardiac"
                    else:
                        value_new = ""
                elif key == "ocarasdsec":
                    value_new = categorize_binary(value)
                # Pick up here to add more
                else:
                    # Replace 'nan' string with empty string
                    if np.isnan(value):
                        value_new = ""
                    # Otherwise, retain the value as is
                    else:
                        value_new = value

                # If value_new is an int, cast to str
                if isinstance(value_new, int) or isinstance(value_new, float):
                    value_new = str(value_new)

                # Append feature string
                if value_new is None:
                    features_str += '"",'
                else:
                    features_str += '"' + value_new + '",'

            # Remove last comma
            features_str = features_str[:-1]

            # Cap end bracket
            features_str += "}"

            if print_features_str:
                print("Features from CSV:")
                print(features_str)

            # Call function to query STS website
            time.sleep(0.250)
            outcomes_string = query_sts_website(features_str)

            # Format the JSON text into dict
            outcomes[row] = json.loads(outcomes_string)

            # Loop through dict and convert strings of outcomes
            # into floats and convert to percentage
            # and save values to the row'th row in the dict
            for key, value in outcomes[row].items():
                if value is None:
                    outcomes[row][key] = "NA"
                else:
                    outcomes[row][key] = "%2.3f%%" % (float(value) * 100)

            print("\n")
            print("Risk of Mortality: %s" % outcomes[row]["predmort"])
            print("Renal Failure: %s" % outcomes[row]["predrenf"])
            print("Permanent Stroke: %s" % outcomes[row]["predstro"])
            print("Prolonged Ventilation: %s" % outcomes[row]["predvent"])
            print("DSW Infection: %s" % outcomes[row]["preddeep"])
            print("Reoperation: %s" % outcomes[row]["predreop"])
            print("Morbidity or Mortality: %s" % outcomes[row]["predmm"])
            print("Short Length of Stay: %s" % outcomes[row]["pred6d"])
            print("Long Length of Stay: %s" % outcomes[row]["pred14d"])
            print("\n")

    # Convert dict of outcomes to df
    outcomes_df = pd.DataFrame.from_dict(outcomes, orient="index")

    # Create HDFStore object in current working directory
    # ./cardiac-surgery/
    store = pd.HDFStore("outcomes_df.h5")

    # Save outcomes dataframe in H5 as 'df'
    store["df"] = outcomes_df

    print("Successfully saved outcomes dataframe to H5 file!")
