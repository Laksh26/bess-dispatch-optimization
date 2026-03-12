import json

import pandas as pd

from pathlib import Path
from benchopt import BaseDataset


class LinearMILPDataset(BaseDataset):

    name = "linear_milp_bess_database"

    requirements = ["pandas", "numpy"]

    parameters = {}

    def get_data(self):
        ROOT_DIR = Path(__file__).resolve().parent.parent
        config_path = ROOT_DIR / "RUN_CONFIG.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # resolution of the forecasts provided as input
        resolution = float(int(config["resolution_min"]) / 60)  # hour

        renewable_feed_in_tariff = (
            float(config["renewable_feed_in_tariff_cents_per_kWh"]) / 1e3
        )  # cents per Watt hour

        dataset_path = ROOT_DIR / "Datasource"
        data_points_for_opti = int(1 * 24 / resolution)  # 1day
        print(
            f"Optimizing for 1 day with {resolution} min resolution = {data_points_for_opti} points"
        )
        csv_path = dataset_path / config["spot_price_eur_per_MWh_file_path"]
        df = pd.read_csv(csv_path, sep=";")
        df = df[:data_points_for_opti]
        price_eur_per_MWh = (
            df["Day-ahead Price (EUR/MWh)"].astype(float).to_numpy()
            * config["scale_spot_price"]
        )
        spot_price = price_eur_per_MWh * (100 / 1e6)  # cents per Watt hour

        T = len(spot_price)

        # load pv generation
        csv_path = dataset_path / config["load_forecast_W_file_path"]
        df = pd.read_csv(csv_path)
        df = df.head(T)
        load_forecast = (
            df["Electrical_Load"].astype(float).to_numpy()
            * config["scale_load_forecast"]
        )

        csv_path = dataset_path / config["renewable_generation_forecast_W_file_path"]
        df = pd.read_csv(csv_path)
        df = df.head(T)
        renewable_generation_forecast = (
            df["Renewable_generation"].astype(float).to_numpy()
            * config["scale_renewable_generation"]
        )

        if not (len(load_forecast) == len(renewable_generation_forecast) == T):
            raise ValueError("Input time series length mismatch.")

        # Nominal power of the battery inverter
        P_nom = config["bess_nominal_power_W"]  # W

        # Nominal discharge capacity of the BESS
        E_nom = config["bess_nominal_energy_capacity_Wh"]  # Wh

        # battery internal efficiency (WRTE) in a round trip. Check the protocol - Speichervermessung KIT + HTW Berlin
        batt_eff = config["bess_round_trip_eff"]

        # Inverter efficiency in charging direction
        inv_eff_ch = config["bess_inv_charging_eff"]

        # Inverter efficiency in discharging direction
        inv_eff_dch = config["bess_inv_discharging_eff"]

        # Initialization values
        # Impl note: Cannot model SoC (State of Charge). It requires voltage and current level modeling which is ignored
        # in linear model for simplification.
        #
        # This creates a challenge - Initializing start SoE (State of Energy).
        #
        # As a simplification, one can set this up with the SoC directly. This simplification is acceptable in BESS
        # technologies such as Lithium-ion Batteries, but if the SoC vs SoE dynamics is complex, then a mapping of SoC
        # to SoE is necessary to model the difference correctly.
        soe_start = config["start_soc"]

        data = dict(
            T=T,
            load=load_forecast,
            renewable_generation=renewable_generation_forecast,
            price=spot_price,
            renewable_feed_in_tariff=renewable_feed_in_tariff,
            P_nom=P_nom,
            E_nom=E_nom,
            batt_eff=batt_eff,
            inv_eff_ch=inv_eff_ch,
            inv_eff_dch=inv_eff_dch,
            dt=resolution,
            soe_start=soe_start,
        )

        return data
