import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display


class MatplotlibPatientTrajectory:

    def load_data(self, original_df):
        original_df = original_df
        original_df["episode_start_date"] = pd.to_datetime(
            original_df["episode_start_date"]
        )
        original_df["episode_end_date"] = pd.to_datetime(
            original_df["episode_end_date"], errors="coerce"
        )
        original_df = original_df.sort_values(by=["initial_age", "Length_of_Episode"])
        return original_df

    def patient_timeline_plot_yearly(self, original_df, ax, cmap, unique_patients):
        for i, patient_id in enumerate(unique_patients):
            patient_data = original_df[original_df["pasient"] == patient_id]
            for _, case in patient_data.iterrows():
                case_start = case["episode_start_date"]
                case_end = case["episode_end_date"]
                case_age = case["age"]
                case_cluster = case["cluster"]

                age_start = case_age
                age_end = age_start + (case_end - case_start).days / 365.2425

                ax.plot(
                    [age_start, age_end],
                    [i, i],
                    linewidth=5,
                    color=cmap[case_cluster],
                )

        ax.set_xlim(
            original_df["age"].min(),
            18,
        )
        ax.set_ylim(
            0,
        )
        ax.set_xticks([0, 6, 12, 18])
        for x in [0, 6, 12, 18]:
            ax.axvline(x, color="gray", linewidth=3, linestyle="--")


class MatplotlibPatientTrajectoryMiddleChildhood:

    def load_data(self, original_df):
        original_df = original_df
        original_df["episode_start_date"] = pd.to_datetime(
            original_df["episode_start_date"]
        )
        original_df["episode_end_date"] = pd.to_datetime(
            original_df["episode_end_date"], errors="coerce"
        )
        original_df = original_df.sort_values(by=["initial_age", "Length_of_Episode"])
        return original_df

    def patient_timeline_plot_yearly(self, original_df, ax, cmap, unique_patients):
        for i, patient_id in enumerate(unique_patients):
            patient_data = original_df[original_df["pasient"] == patient_id]
            for _, case in patient_data.iterrows():
                case_start = case["episode_start_date"]
                case_end = case["episode_end_date"]
                case_age = case["age"]
                case_cluster = case["cluster"]

                age_start = case_age
                age_end = age_start + (case_end - case_start).days / 365.2425

                ax.plot(
                    [age_start, age_end],
                    [i, i],
                    linewidth=5,
                    color=cmap[case_cluster],
                )

        ax.set_xlim(
            original_df["age"].min(),
            18,
        )
        ax.set_ylim(
            0,
        )
        ax.set_xticks([0, 6, 12, 18])
        for x in [0, 6, 12, 18]:
            ax.axvline(x, color="gray", linewidth=3, linestyle="--")


class MatplotlibPatientTrajectoryTeenager:

    def load_data(self, original_df):
        original_df = original_df
        original_df["episode_start_date"] = pd.to_datetime(
            original_df["episode_start_date"]
        )
        original_df["episode_end_date"] = pd.to_datetime(
            original_df["episode_end_date"], errors="coerce"
        )
        original_df = original_df.sort_values(by=["initial_age", "Length_of_Episode"])
        return original_df

    def patient_timeline_plot_yearly(self, original_df, ax, cmap, unique_patients):
        for i, patient_id in enumerate(unique_patients):
            patient_data = original_df[original_df["pasient"] == patient_id]
            for _, case in patient_data.iterrows():
                case_start = case["episode_start_date"]
                case_end = case["episode_end_date"]
                case_age = case["age"]
                case_cluster = case["cluster"]

                age_start = case_age
                age_end = age_start + (case_end - case_start).days / 365.2425

                ax.plot(
                    [age_start, age_end],
                    [i, i],
                    linewidth=5,
                    color=cmap[case_cluster],
                )

        ax.set_xlim(
            original_df["age"].min(),
            18,
        )
        ax.set_ylim(
            0,
        )
        ax.set_xticks([0, 6, 12, 18])
        for x in [0, 6, 12, 18]:
            ax.axvline(x, color="gray", linewidth=3, linestyle="--")
