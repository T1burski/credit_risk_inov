import numpy as np
import pandas as pd
from pyspark.sql import functions as F


def feature_engineering(df_raw):
    median_annual_inc = float(
        62500
    )  # value defined in EDA using training data. Will be considered as constant until new analysis.

    df = df_raw.withColumn("earliest_cr_line_date", F.to_date(F.col("earliest_cr_line"), "MMM-yyyy"))

    df = df.withColumn("issue_d_date", F.to_date(F.col("issue_d"), "MMM-yyyy"))

    df = df.withColumn(
        "mths_since_earliest_cr_line",
        F.round(F.months_between(F.lit("2017-12-01").cast("date"), F.col("earliest_cr_line_date"))).cast("int"),
    )

    default_list = [
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
        "Late (31-120 days)",
    ]

    df = df.withColumn("default_bin", F.when(F.col("loan_status").isin(default_list), 1).otherwise(0))

    df = df.withColumn(
        "total_rev_hi_lim",
        F.when(F.col("total_rev_hi_lim").isNull(), F.col("funded_amnt")).otherwise(F.col("total_rev_hi_lim")),
    )

    treat_with_zeros = ["mths_since_earliest_cr_line", "inq_last_6mths"]

    for c in treat_with_zeros:
        df = df.withColumn(c, F.when(F.col(c).isNull(), 0).otherwise(F.col(c)))

    df = df.withColumn(
        "annual_inc", F.when(F.col("annual_inc").isNull(), median_annual_inc).otherwise(F.col("annual_inc"))
    )

    df = df.toPandas()

    def feature_binning(df_in):
        df = df_in.copy()

        # ─────────────────────────────────────────────
        # NUMERICAL FEATURES
        # ─────────────────────────────────────────────

        # dti
        df["dti"] = pd.cut(
            df["dti"],
            bins=[-np.inf, 8.89, 10.30, 12.14, 13.38, 14.53, 15.59, 17.13, 18.80, 20.18, 21.44, 25.85, 29.01, np.inf],
            labels=[
                "(-inf, 8.89)",
                "[8.89, 10.30)",
                "[10.30, 12.14)",
                "[12.14, 13.38)",
                "[13.38, 14.53)",
                "[14.53, 15.59)",
                "[15.59, 17.13)",
                "[17.13, 18.80)",
                "[18.80, 20.18)",
                "[20.18, 21.44)",
                "[21.44, 25.85)",
                "[25.85, 29.01)",
                "[29.01, inf)",
            ],
        )

        # annual_inc
        df["annual_inc"] = pd.cut(
            df["annual_inc"],
            bins=[
                -np.inf,
                28338.00,
                37086.20,
                40371.00,
                49464.10,
                60996.50,
                66098.00,
                70703.40,
                80046.22,
                90230.50,
                100131.00,
                120012.00,
                np.inf,
            ],
            labels=[
                "(-inf, 28338.00)",
                "[28338.00, 37086.20)",
                "[37086.20, 40371.00)",
                "[40371.00, 49464.10)",
                "[49464.10, 60996.50)",
                "[60996.50, 66098.00)",
                "[66098.00, 70703.40)",
                "[70703.40, 80046.22)",
                "[80046.22, 90230.50)",
                "[90230.50, 100131.00)",
                "[100131.00, 120012.00)",
                "[120012.00, inf)",
            ],
        )

        # int_rate
        df["int_rate"] = pd.cut(
            df["int_rate"],
            bins=[-np.inf, 7.74, 8.92, 10.15, 11.01, 12.01, 13.05, 13.98, 15.12, 15.61, 17.57, 19.01, 20.99, np.inf],
            labels=[
                "(-inf, 7.74)",
                "[7.74, 8.92)",
                "[8.92, 10.15)",
                "[10.15, 11.01)",
                "[11.01, 12.01)",
                "[12.01, 13.05)",
                "[13.05, 13.98)",
                "[13.98, 15.12)",
                "[15.12, 15.61)",
                "[15.61, 17.57)",
                "[17.57, 19.01)",
                "[19.01, 20.99)",
                "[20.99, inf)",
            ],
        )

        # total_rev_hi_lim
        df["total_rev_hi_lim"] = pd.cut(
            df["total_rev_hi_lim"],
            bins=[-np.inf, 5940.50, 11726.00, 20294.50, 28118.50, 36034.50, 44662.00, 55862.00, np.inf],
            labels=[
                "(-inf, 5940.50)",
                "[5940.50, 11726.00)",
                "[11726.00, 20294.50)",
                "[20294.50, 28118.50)",
                "[28118.50, 36034.50)",
                "[36034.50, 44662.00)",
                "[44662.00, 55862.00)",
                "[55862.00, inf)",
            ],
        )

        # inq_last_6mths
        df["inq_last_6mths"] = pd.cut(
            df["inq_last_6mths"],
            bins=[-np.inf, 0.50, 1.50, 2.50, np.inf],
            labels=["(-inf, 0.50)", "[0.50, 1.50)", "[1.50, 2.50)", "[2.50, inf)"],
        )

        # mths_since_earliest_cr_line
        df["mths_since_earliest_cr_line"] = pd.cut(
            df["mths_since_earliest_cr_line"],
            bins=[-np.inf, 145.50, 168.50, 204.50, 228.50, 247.50, 266.50, 287.50, 353.50, np.inf],
            labels=[
                "(-inf, 145.50)",
                "[145.50, 168.50)",
                "[168.50, 204.50)",
                "[204.50, 228.50)",
                "[228.50, 247.50)",
                "[247.50, 266.50)",
                "[266.50, 287.50)",
                "[287.50, 353.50)",
                "[353.50, inf)",
            ],
        )

        # ─────────────────────────────────────────────
        # CATEGORICAL FEATURES
        # ─────────────────────────────────────────────

        # purpose
        purpose_map = {
            "credit_card": "['credit_card', 'car']",
            "car": "['credit_card', 'car']",
            "major_purchase": "['major_purchase', 'home_improvement']",
            "home_improvement": "['major_purchase', 'home_improvement']",
            "wedding": "['wedding', 'debt_consolidation', 'vacation']",
            "debt_consolidation": "['wedding', 'debt_consolidation', 'vacation']",
            "vacation": "['wedding', 'debt_consolidation', 'vacation']",
            "medical": "['medical', 'house', 'other', 'moving', 'renewable_energy', 'educational', 'small_business']",
            "house": "['medical', 'house', 'other', 'moving', 'renewable_energy', 'educational', 'small_business']",
            "other": "['medical', 'house', 'other', 'moving', 'renewable_energy', 'educational', 'small_business']",
            "moving": "['medical', 'house', 'other', 'moving', 'renewable_energy', 'educational', 'small_business']",
            "renewable_energy": "['medical', 'house', 'other', 'moving', 'renewable_energy', 'educational', 'small_business']",
            "small_business": "['medical', 'house', 'other', 'moving', 'renewable_energy', 'educational', 'small_business']",
            "educational": "['medical', 'house', 'other', 'moving', 'renewable_energy', 'educational', 'small_business']",
        }
        df["purpose"] = (
            df["purpose"]
            .map(purpose_map)
            .fillna("['medical', 'house', 'other', 'moving', 'renewable_energy', 'educational', 'small_business']")
        )  # fill NaN with the worst case scenario

        # initial_list_status
        df["initial_list_status"] = (
            df["initial_list_status"]
            .map(
                {
                    "w": "['w']",
                    "f": "['f']",
                }
            )
            .fillna("['f']")
        )  # fill NaN with the worst case scenario

        # verification_status
        df["verification_status"] = (
            df["verification_status"]
            .map(
                {
                    "Not Verified": "['Not Verified']",
                    "Source Verified": "['Source Verified']",
                    "Verified": "['Verified']",
                }
            )
            .fillna("['Verified']")
        )  # fill NaN with the worst case scenario

        # addr_state
        addr_state_map = {
            # Bin 1
            "ME": "['ME', 'DC', 'WY', 'ID', 'NH', 'WV', 'AK', 'CO', 'KS', 'MS']",
            "DC": "['ME', 'DC', 'WY', 'ID', 'NH', 'WV', 'AK', 'CO', 'KS', 'MS']",
            "WY": "['ME', 'DC', 'WY', 'ID', 'NH', 'WV', 'AK', 'CO', 'KS', 'MS']",
            "ID": "['ME', 'DC', 'WY', 'ID', 'NH', 'WV', 'AK', 'CO', 'KS', 'MS']",
            "NH": "['ME', 'DC', 'WY', 'ID', 'NH', 'WV', 'AK', 'CO', 'KS', 'MS']",
            "WV": "['ME', 'DC', 'WY', 'ID', 'NH', 'WV', 'AK', 'CO', 'KS', 'MS']",
            "AK": "['ME', 'DC', 'WY', 'ID', 'NH', 'WV', 'AK', 'CO', 'KS', 'MS']",
            "CO": "['ME', 'DC', 'WY', 'ID', 'NH', 'WV', 'AK', 'CO', 'KS', 'MS']",
            "KS": "['ME', 'DC', 'WY', 'ID', 'NH', 'WV', 'AK', 'CO', 'KS', 'MS']",
            "MS": "['ME', 'DC', 'WY', 'ID', 'NH', 'WV', 'AK', 'CO', 'KS', 'MS']",
            # Bin 2
            "MT": "['MT', 'VT', 'SC', 'TX']",
            "VT": "['MT', 'VT', 'SC', 'TX']",
            "SC": "['MT', 'VT', 'SC', 'TX']",
            "TX": "['MT', 'VT', 'SC', 'TX']",
            # Bin 3
            "CT": "['CT', 'IL']",
            "IL": "['CT', 'IL']",
            # Bin 4
            "OR": "['OR', 'WI', 'WA', 'MN', 'GA', 'SD']",
            "WI": "['OR', 'WI', 'WA', 'MN', 'GA', 'SD']",
            "WA": "['OR', 'WI', 'WA', 'MN', 'GA', 'SD']",
            "MN": "['OR', 'WI', 'WA', 'MN', 'GA', 'SD']",
            "GA": "['OR', 'WI', 'WA', 'MN', 'GA', 'SD']",
            "SD": "['OR', 'WI', 'WA', 'MN', 'GA', 'SD']",
            # Bin 5
            "DE": "['DE', 'MA', 'IN', 'KY', 'RI']",
            "MA": "['DE', 'MA', 'IN', 'KY', 'RI']",
            "IN": "['DE', 'MA', 'IN', 'KY', 'RI']",
            "KY": "['DE', 'MA', 'IN', 'KY', 'RI']",
            "RI": "['DE', 'MA', 'IN', 'KY', 'RI']",
            # Bin 6
            "OH": "['OH', 'PA', 'AZ', 'LA']",
            "PA": "['OH', 'PA', 'AZ', 'LA']",
            "AZ": "['OH', 'PA', 'AZ', 'LA']",
            "LA": "['OH', 'PA', 'AZ', 'LA']",
            # Bin 7
            "UT": "['UT', 'MI', 'VA']",
            "MI": "['UT', 'MI', 'VA']",
            "VA": "['UT', 'MI', 'VA']",
            # Bin 8
            "CA": "['CA', 'TN', 'AR']",
            "TN": "['CA', 'TN', 'AR']",
            "AR": "['CA', 'TN', 'AR']",
            # Bin 9
            "NC": "['NC', 'OK', 'MD']",
            "OK": "['NC', 'OK', 'MD']",
            "MD": "['NC', 'OK', 'MD']",
            # Bin 10
            "MO": "['MO', 'NY', 'NJ', 'NM']",
            "NY": "['MO', 'NY', 'NJ', 'NM']",
            "NJ": "['MO', 'NY', 'NJ', 'NM']",
            "NM": "['MO', 'NY', 'NJ', 'NM']",
            # Bin 11
            "AL": "['AL', 'HI', 'FL', 'NV', 'IA', 'NE']",
            "HI": "['AL', 'HI', 'FL', 'NV', 'IA', 'NE']",
            "FL": "['AL', 'HI', 'FL', 'NV', 'IA', 'NE']",
            "NV": "['AL', 'HI', 'FL', 'NV', 'IA', 'NE']",
            "IA": "['AL', 'HI', 'FL', 'NV', 'IA', 'NE']",
            "NE": "['AL', 'HI', 'FL', 'NV', 'IA', 'NE']",
        }
        df["addr_state"] = (
            df["addr_state"].map(addr_state_map).fillna("['AL', 'HI', 'FL', 'NV', 'IA', 'NE']")
        )  # fill NaN with the worst case scenario

        # home_ownership
        df["home_ownership"] = (
            df["home_ownership"]
            .map(
                {
                    "MORTGAGE": "['MORTGAGE']",
                    "OWN": "['OWN']",
                    "RENT": "['RENT', 'NONE', 'OTHER']",
                    "NONE": "['RENT', 'NONE', 'OTHER']",
                    "OTHER": "['RENT', 'NONE', 'OTHER']",
                }
            )
            .fillna("['RENT', 'NONE', 'OTHER']")
        )  # fill NaN with the worst case scenario

        # sub_grade
        last_bin = "['E2', 'E3', 'E4', 'F1', 'E5', 'F2', 'G4', 'F3', 'F4', 'G2', 'G5', 'G3', 'F5', 'G1']"
        sub_grade_map = {
            "A1": "['A1', 'A2', 'A3']",
            "A2": "['A1', 'A2', 'A3']",
            "A3": "['A1', 'A2', 'A3']",
            "A4": "['A4', 'A5']",
            "A5": "['A4', 'A5']",
            "B1": "['B1']",
            "B2": "['B2']",
            "B3": "['B3']",
            "B4": "['B4']",
            "B5": "['B5']",
            "C1": "['C1']",
            "C2": "['C2']",
            "C3": "['C3']",
            "C4": "['C4']",
            "C5": "['C5', 'D1']",
            "D1": "['C5', 'D1']",
            "D2": "['D2', 'D3']",
            "D3": "['D2', 'D3']",
            "D4": "['D4', 'D5', 'E1']",
            "D5": "['D4', 'D5', 'E1']",
            "E1": "['D4', 'D5', 'E1']",
            "E2": last_bin,
            "E3": last_bin,
            "E4": last_bin,
            "E5": last_bin,
            "F1": last_bin,
            "F2": last_bin,
            "F3": last_bin,
            "F4": last_bin,
            "F5": last_bin,
            "G1": last_bin,
            "G2": last_bin,
            "G3": last_bin,
            "G4": last_bin,
            "G5": last_bin,
        }
        df["sub_grade"] = (
            df["sub_grade"]
            .map(sub_grade_map)
            .fillna("['E2', 'E3', 'E4', 'F1', 'E5', 'F2', 'G4', 'F3', 'F4', 'G2', 'G5', 'G3', 'F5', 'G1']")
        )  # fill NaN with the worst case scenario

        return df

    return feature_binning(df)
