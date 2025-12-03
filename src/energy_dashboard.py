import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Paths & logging
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=OUTPUT_DIR / "energy_dashboard.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ----------------------------
# Task 3: OOP modelling
# ----------------------------

@dataclass
class MeterReading:
    timestamp: pd.Timestamp
    kwh: float


class Building:
    def __init__(self, name: str):
        self.name = name
        self.meter_readings: list[MeterReading] = []

    def add_reading(self, reading: MeterReading) -> None:
        self.meter_readings.append(reading)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": [r.timestamp for r in self.meter_readings],
                "kwh": [r.kwh for r in self.meter_readings],
                "building": self.name,
            }
        )

    def calculate_total_consumption(self) -> float:
        return sum(r.kwh for r in self.meter_readings)

    def generate_report(self) -> str:
        df = self.to_dataframe()
        total = df["kwh"].sum()
        mean = df["kwh"].mean()
        max_v = df["kwh"].max()
        min_v = df["kwh"].min()
        return (
            f"Building: {self.name}\n"
            f"  Total kWh: {total:.2f}\n"
            f"  Mean kWh : {mean:.2f}\n"
            f"  Max kWh  : {max_v:.2f}\n"
            f"  Min kWh  : {min_v:.2f}\n"
        )


class BuildingManager:
    def __init__(self):
        self.buildings: dict[str, Building] = {}

    def get_or_create_building(self, name: str) -> Building:
        if name not in self.buildings:
            self.buildings[name] = Building(name)
        return self.buildings[name]

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            b = self.get_or_create_building(row["building"])
            reading = MeterReading(timestamp=row["timestamp"], kwh=row["kwh"])
            b.add_reading(reading)

    def campus_total_consumption(self) -> float:
        return sum(b.calculate_total_consumption() for b in self.buildings.values())

    def highest_consuming_building(self) -> Building:
        return max(self.buildings.values(), key=lambda b: b.calculate_total_consumption())


# ----------------------------
# Task 1: Data ingestion
# ----------------------------

def load_and_validate_data(data_dir: Path) -> pd.DataFrame:
    logging.info(f"Loading CSV files from {data_dir}")
    frames = []

    for csv_file in data_dir.glob("*.csv"):
        try:
            logging.info(f"Reading file: {csv_file.name}")
            df = pd.read_csv(csv_file, on_bad_lines="skip")

            if "timestamp" not in df.columns or "kwh" not in df.columns:
                logging.error(f"File {csv_file.name} missing required columns.")
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp", "kwh"])

            if "building" not in df.columns:
                df["building"] = csv_file.stem  

            frames.append(df)

        except FileNotFoundError:
            logging.error(f"File not found: {csv_file}")
        except Exception as e:
            logging.error(f"Error reading {csv_file}: {e}")

    if not frames:
        raise ValueError("No valid CSV files found in data directory.")

    df_combined = pd.concat(frames, ignore_index=True)
    df_combined.sort_values("timestamp", inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    logging.info(f"Combined data shape: {df_combined.shape}")
    return df_combined


# ----------------------------
# Task 2: Aggregation logic
# ----------------------------

def calculate_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp")
    daily = df.resample("D")["kwh"].sum().reset_index()
    daily.rename(columns={"kwh": "daily_kwh"}, inplace=True)
    return daily


def calculate_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp")
    weekly = df.resample("W")["kwh"].sum().reset_index()
    weekly.rename(columns={"kwh": "weekly_kwh"}, inplace=True)
    return weekly


def building_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("building")["kwh"]
        .agg(["mean", "min", "max", "sum"])
        .rename(columns={"sum": "total_kwh"})
        .reset_index()
    )
    return summary


# ----------------------------
# Task 4: Visualizations
# ----------------------------

def create_dashboard_plots(df: pd.DataFrame, output_dir: Path) -> None:
    daily = (
        df.set_index("timestamp")
        .groupby("building")["kwh"]
        .resample("D")
        .sum()
        .reset_index()
    )

    weekly = (
        df.set_index("timestamp")
        .groupby("building")["kwh"]
        .resample("W")
        .sum()
        .groupby("building")
        .mean()
        .reset_index()
    )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    for b_name, group in daily.groupby("building"):
        ax1.plot(group["timestamp"], group["kwh"], label=b_name)
    ax1.set_title("Daily Energy Consumption Trend")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("kWh")
    ax1.legend()
    ax1.grid(True)

    ax2.bar(weekly["building"], weekly["kwh"])
    ax2.set_title("Average Weekly Consumption by Building")
    ax2.set_xlabel("Building")
    ax2.set_ylabel("kWh (weekly average)")

    ax3.scatter(df["timestamp"], df["kwh"])
    ax3.set_title("Consumption Scatter (All Readings)")
    ax3.set_xlabel("Timestamp")
    ax3.set_ylabel("kWh")

    plt.tight_layout()
    dashboard_path = output_dir / "dashboard.png"
    plt.savefig(dashboard_path)
    plt.close()
    logging.info(f"Dashboard saved to {dashboard_path}")


# ----------------------------
# Task 5: Persistence & summary
# ----------------------------

def save_outputs(
    df_clean: pd.DataFrame,
    building_summary_df: pd.DataFrame,
    manager: BuildingManager,
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    cleaned_path = output_dir / "cleaned_energy_data.csv"
    df_clean.to_csv(cleaned_path, index=False)

    summary_path = output_dir / "building_summary.csv"
    building_summary_df.to_csv(summary_path, index=False)

    campus_total = manager.campus_total_consumption()
    top_building = manager.highest_consuming_building()

    peak_row = df_clean.loc[df_clean["kwh"].idxmax()]
    peak_time = peak_row["timestamp"]
    peak_building = peak_row["building"]
    peak_value = peak_row["kwh"]

    weekly_mean = weekly_df["weekly_kwh"].mean()
    daily_mean = daily_df["daily_kwh"].mean()

    summary_lines = [
        "Campus Energy Use – Executive Summary\n",
        f"Total campus consumption: {campus_total:.2f} kWh",
        f"Highest-consuming building: {top_building.name} "
        f"({top_building.calculate_total_consumption():.2f} kWh)",
        f"Peak load time: {peak_time} in {peak_building} "
        f"({peak_value:.2f} kWh)",
        f"Average daily consumption (campus): {daily_mean:.2f} kWh",
        f"Average weekly consumption (campus): {weekly_mean:.2f} kWh",
    ]

    summary_path_txt = output_dir / "summary.txt"
    with open(summary_path_txt, "w") as f:
        f.write("\n".join(summary_lines))

    print("\n".join(summary_lines))
    logging.info(f"Summary written to {summary_path_txt}")


# ----------------------------
# Main pipeline
# ----------------------------

def main():
    df = load_and_validate_data(DATA_DIR)

    daily_totals = calculate_daily_totals(df)
    weekly_totals = calculate_weekly_aggregates(df)
    b_summary = building_wise_summary(df)

    manager = BuildingManager()
    df_for_oop = df.copy()
    df_for_oop["timestamp"] = pd.to_datetime(df_for_oop["timestamp"])
    manager.load_from_dataframe(df_for_oop)

    create_dashboard_plots(df, OUTPUT_DIR)

    save_outputs(
        df_clean=df,
        building_summary_df=b_summary,
        manager=manager,
        daily_df=daily_totals,
        weekly_df=weekly_totals,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
