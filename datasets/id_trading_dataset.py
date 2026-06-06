import json
import numpy as np

from pathlib import Path
from benchopt import BaseDataset


class IDTradingDataset(BaseDataset):

    name = "id_trading_bess_dataset"
    requirements = ["numpy"]
    parameters = {}

    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path

    def get_data(self):

        # ======================== gather the ESS config ========================
        ess_config_path = Path(self.config_path) / "ESS_CONFIG.json"
        with open(ess_config_path, "r") as f:
            ess_config = json.load(f)

        E_nom = []
        batt_eff = []
        P_nom = []
        minOperationPower = []
        inv_eff_ch = []
        inv_eff_dch = []

        ess_count = int(ess_config["ess_count"])
        if ess_count == 0:
            raise ValueError("No ess count provided.")

        for i in range(ess_count):
            ess_name = f"ess{i + 1}"
            E_nom.append(float(ess_config["ess_details"][ess_name]["bess_nominal_energy_capacity_Wh"]))
            batt_eff.append(float(ess_config["ess_details"][ess_name]["round_trip_eff_at_half_nom_power"]))
            P_nom.append(float(ess_config["ess_details"][ess_name]["inv_nominal_power_W"]))
            minOperationPower.append(float(ess_config["ess_details"][ess_name]["minimum_operation_power_W"]))
            inv_eff_ch.append(float(ess_config["ess_details"][ess_name]["inv_charging_eff"]))
            inv_eff_dch.append(float(ess_config["ess_details"][ess_name]["inv_discharging_eff"]))

        pwl = ess_config["pwl_efficiency_fits"]
        n_pieces = int(pwl["n_pieces"])
        inv_ch_coeffs = list(pwl["inv_ch_loss_fit_W_per_kW"])
        inv_dch_coeffs = list(pwl["inv_dch_loss_fit_W_per_kW"])
        batt_coeffs = list(pwl["batt_loss_fit_W_per_kW"])

        # ======================== gather the ID config ========================
        id_config_path = Path(self.config_path) / "ID_CONFIG.json"
        with open(id_config_path, "r") as f:
            id_config = json.load(f)

        isEssAvailableForOperation = []
        soe_start = []
        temp_start = []
        for i in range(ess_count):
            ess_name = f"ess{i + 1}"
            isEssAvailableForOperation.append(id_config["ess_states"][ess_name]["isAvailable"])
            if not isEssAvailableForOperation[i]:
                P_nom[i] = 0.0
            soe_start.append(float(id_config["ess_states"][ess_name]["start_soc_0_to_1"]))
            temp_start.append(float(id_config["ess_states"][ess_name]["start_temperature_degree_C"]))

        resolution = float(int(id_config["resolution_min"]) / 60)

        price_raw = (
            np.array(id_config["forecasts"]["spot_price_EUR_per_MWh"])
            * float(id_config["forecast_scalers"]["spot_price_EUR_per_MW"])
        )
        spot_price = price_raw * (100 / 1e6)

        T = len(spot_price)

        return dict(
            N_bess=ess_count,
            T=T,
            dt=resolution,
            load=np.zeros(T),
            renewable_generation=np.zeros(T),
            price=spot_price,
            renewable_feed_in_tariff=0.0,
            scenario="trading",
            is_ess_available_for_operation=isEssAvailableForOperation,
            P_nom=P_nom,
            min_operation_power=minOperationPower,
            E_nom=E_nom,
            batt_eff=batt_eff,
            inv_eff_ch=inv_eff_ch,
            inv_eff_dch=inv_eff_dch,
            soe_start=soe_start,
            temp_start=temp_start,
            inv_ch_coeffs=inv_ch_coeffs,
            inv_dch_coeffs=inv_dch_coeffs,
            batt_coeffs=batt_coeffs,
            n_pieces=n_pieces,
        )