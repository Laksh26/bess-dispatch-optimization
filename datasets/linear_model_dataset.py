import json

import numpy as np
import pandas as pd

from pathlib import Path
from benchopt import BaseDataset


class LinearMILPDataset(BaseDataset):

    name = "linear_milp_bess_database"

    requirements = ["pandas", "numpy"]

    parameters = {}

    def get_data(self):
        ROOT_DIR = Path(__file__).resolve().parent.parent

        # ======================== gather the ESS config ========================
        config_path = ROOT_DIR / "ESS_CONFIG.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Bess intrinsic properties
        E_nom = []
        batt_eff = []

        # BESS hardware properties
        P_nom = []
        minOperationPower = []
        inv_eff_ch = []
        inv_eff_dch = []

        ess_count = int(config["ess_count"])
        if ess_count == 0:
            raise ValueError("No ess count provided.")

        for i in range(ess_count):
            ess_name = f"ess{i + 1}"
            E_nom.append(
                float(
                    config["ess_details"][ess_name]["bess_nominal_energy_capacity_Wh"]
                )
            )
            batt_eff.append(
                float(
                    config["ess_details"][ess_name]["round_trip_eff_at_half_nom_power"]
                )
            )

            P_nom.append(float(config["ess_details"][ess_name]["inv_nominal_power_W"]))
            minOperationPower.append(
                float(config["ess_details"][ess_name]["minimum_operation_power_W"])
            )
            inv_eff_ch.append(
                float(config["ess_details"][ess_name]["inv_charging_eff"])
            )
            inv_eff_dch.append(
                float(config["ess_details"][ess_name]["inv_discharging_eff"])
            )

        # ======================== gather the RUN config ========================
        config_path = ROOT_DIR / "RUN_CONFIG.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        isEssAvailableForOperation = []
        soe_start = []
        temp_start = []  # currently unused. Planned for more complex models
        for i in range(ess_count):
            ess_name = f"ess{i + 1}"
            isEssAvailableForOperation.append(
                config["ess_states"][ess_name]["isAvailable"]
            )
            if not isEssAvailableForOperation[i]:
                P_nom[i] = 0.0
            # Impl note: Cannot model SoC (State of Charge). It requires voltage and current level modeling which is
            # ignored in linear model for simplification.
            #
            # This creates a challenge - Initializing start SoE (State of Energy).
            #
            # As a simplification, one can set this up with the SoC directly. This simplification is acceptable in BESS
            # technologies such as Lithium-ion Batteries, but if the SoC vs SoE dynamics is complex, then a mapping of
            # SoC to SoE is necessary to model the difference correctly.
            soe_start.append(float(config["ess_states"][ess_name]["start_soc_0_to_1"]))
            temp_start.append(
                float(config["ess_states"][ess_name]["start_temperature_degree_C"])
            )

        # resolution of the forecasts provided as input
        resolution = float(int(config["resolution_min"]) / 60)  # hour

        renewable_feed_in_tariff = (
            float(config["renewable_feed_in_tariff_cents_per_kWh"]) / 1e3
        )  # cents per Watt hour

        renewable_generation_forecast = (
            np.array(config["forecasts"]["renewable_generation_W"])
        ) * float(config["forecast_scalers"]["renewable_generation_W"])

        load_forecast = np.array((config["forecasts"]["load_W"])) * float(
            config["forecast_scalers"]["load_W"]
        )

        price_eur_per_MWh = (
            np.array(config["forecasts"]["spot_price_EUR_per_kWh"])
        ) * float(config["forecast_scalers"]["spot_price_EUR_per_MW"])
        spot_price = price_eur_per_MWh * (100 / 1e6)  # cents per Watt hour

        T = len(spot_price)

        if not len(renewable_generation_forecast) == len(load_forecast) == T:
            raise ValueError("Input time series length mismatch.")

        data = dict(
            N_bess=ess_count,
            T=T,
            dt=resolution,
            load=load_forecast,
            renewable_generation=renewable_generation_forecast,
            price=spot_price,
            renewable_feed_in_tariff=renewable_feed_in_tariff,
            is_ess_available_for_operation=isEssAvailableForOperation,
            P_nom=P_nom,
            min_operation_power=minOperationPower,
            E_nom=E_nom,
            batt_eff=batt_eff,
            inv_eff_ch=inv_eff_ch,
            inv_eff_dch=inv_eff_dch,
            soe_start=soe_start,
            temp_start=temp_start,
        )

        return data
