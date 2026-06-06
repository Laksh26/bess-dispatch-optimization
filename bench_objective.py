from benchopt import BaseObjective
import numpy as np


class Objective(BaseObjective):
    def get_one_result(self):
        return dict(
            p_grid_sale=np.zeros(self.T),
            p_grid_purchase=np.zeros(self.T),
            p_ch=np.zeros(self.N_bess, self.T),
            p_dch=np.zeros(self.N_bess, self.T),
            soe=np.zeros(self.N_bess, self.T),
        )

    name = "BESS dispatch objective"
    parameters = {}

    def set_data(
        self,
        N_bess,
        T,
        dt,
        load,
        renewable_generation,
        price,
        renewable_feed_in_tariff,
        is_ess_available_for_operation,
        P_nom,
        min_operation_power,
        E_nom,
        batt_eff,
        inv_eff_ch,
        inv_eff_dch,
        soe_start,
        temp_start,
        inv_ch_coeffs=None,
        inv_dch_coeffs=None,
        batt_coeffs=None,
        n_pieces=None,
        scenario="self_consumption",
    ):
        self.N_bess = N_bess

        self.T = T
        self.dt = dt

        self.load = load
        self.renewable_generation = renewable_generation
        self.price = price
        self.renewable_feed_in_tariff = renewable_feed_in_tariff

        self.isEssAvailableForOperation = is_ess_available_for_operation
        self.P_nom = P_nom
        self.min_operation_power = min_operation_power
        self.E_nom = E_nom
        self.batt_eff = batt_eff
        self.inv_eff_ch = inv_eff_ch
        self.inv_eff_dch = inv_eff_dch

        self.soe_start = soe_start
        self.temp_start = temp_start

        self.inv_ch_coeffs = inv_ch_coeffs
        self.inv_dch_coeffs = inv_dch_coeffs
        self.batt_coeffs = batt_coeffs
        self.n_pieces = n_pieces
        self.scenario = scenario

    def evaluate_result(self, p_grid_sale, p_grid_purchase, p_ch, p_dch, soe):
        if self.scenario == "trading":
            # Symmetric spot price for both buy and sell; no renewable cap.
            # Negative value = net profit from arbitrage.
            net_cost = np.sum((p_grid_purchase - p_grid_sale) * self.price * self.dt)
            return dict(value=net_cost)

        # self_consumption: sales are capped at renewable generation and paid at feed-in tariff
        p_grid_sale_capped = np.minimum(p_grid_sale, self.renewable_generation)
        feed_in_revenue = p_grid_sale_capped * self.renewable_feed_in_tariff * self.dt
        purchase_cost = p_grid_purchase * self.price * self.dt
        return dict(value=np.sum(purchase_cost) - np.sum(feed_in_revenue))

    def get_objective(self):
        return dict(
            T=self.T,
            N_bess=self.N_bess,
            dt=self.dt,
            load=self.load,
            renewable_generation=self.renewable_generation,
            price=self.price,
            renewable_feed_in_tariff=self.renewable_feed_in_tariff,
            is_ess_available_for_operation=self.isEssAvailableForOperation,
            P_nom=self.P_nom,
            min_operation_power=self.min_operation_power,
            E_nom=self.E_nom,
            batt_eff=self.batt_eff,
            inv_eff_ch=self.inv_eff_ch,
            inv_eff_dch=self.inv_eff_dch,
            soe_start=self.soe_start,
            temp_start=self.temp_start,
            inv_ch_coeffs=self.inv_ch_coeffs,
            inv_dch_coeffs=self.inv_dch_coeffs,
            batt_coeffs=self.batt_coeffs,
            n_pieces=self.n_pieces,
            scenario=self.scenario,
        )
