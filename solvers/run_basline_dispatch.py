import numpy as np
from benchopt import BaseSolver


class BaselineSolver(BaseSolver):

    name = "Baseline solver"

    parameters = {}

    def set_objective(
        self,
        T,
        price,
        load,
        renewable_generation,
        renewable_feed_in_tariff,
        P_nom,
        E_nom,
        batt_eff,
        inv_eff_ch,
        inv_eff_dch,
        dt,
        soe_start,
    ):

        self.T = range(T)
        self.load = load
        self.renewable_generation = renewable_generation
        self.price = price
        self.renewable_feed_in_tariff = renewable_feed_in_tariff
        self.P_nom = P_nom
        self.E_nom = E_nom
        self.batt_eff = batt_eff
        self.inv_eff_ch = inv_eff_ch
        self.inv_eff_dch = inv_eff_dch
        self.dt = dt
        self.soe_start = soe_start

    def run(self, n_iter=1):
        self.p_grid_sale = np.zeros_like(self.T)
        self.p_grid_purchase = np.zeros_like(self.T)
        self.p_ch = np.zeros_like(self.T)
        self.p_dch = np.zeros_like(self.T)
        self.soe = np.zeros_like(self.T)

        for t in self.T:

            surplus = self.load[t] - self.renewable_generation[t]
            if t == 0:
                prev_soe = self.soe_start
            else:
                prev_soe = self.soe[t - 1]

            self.p_ch[t], self.p_dch[t], self.soe[t] = self.apply_bess_power(
                prev_soe=prev_soe, power=surplus
            )

            self.p_grid_sale[t], self.p_grid_purchase[t] = self.apply_grid_power(
                surplus=surplus, p_ch=self.p_ch[t], p_dch=self.p_dch[t]
            )

    def get_result(self):

        return dict(
            p_grid_sale=self.p_grid_sale,
            p_grid_purchase=self.p_grid_purchase,
            p_ch=self.p_ch,
            p_dch=self.p_dch,
            soe=self.soe,
        )

    def apply_grid_power(self, surplus, p_ch, p_dch):
        remaining_surplus = surplus + p_ch - p_dch
        if remaining_surplus < 0:
            p_grid_sale = -remaining_surplus
            p_grid_purchase = 0
        else:
            p_grid_sale = 0
            p_grid_purchase = remaining_surplus

        return p_grid_sale, p_grid_purchase

    def apply_bess_power(self, prev_soe, power):
        sqrt_eff = np.sqrt(self.batt_eff)
        alpha_ch = 1 - (1 - self.inv_eff_ch) - (1 - sqrt_eff)
        alpha_dch = 1 + (1 - self.inv_eff_dch) + (1 - sqrt_eff)

        # max allowed by SoE
        p_ch_max_soe = (1 - prev_soe) * self.E_nom / (alpha_ch * self.dt)
        p_dch_max_soe = prev_soe * self.E_nom / (alpha_dch * self.dt)

        power = np.clip(power, -self.P_nom, self.P_nom)
        if power < 0:  # charging
            p_ch = min(-power, self.P_nom, p_ch_max_soe)
            p_dch = 0

        else:  # discharging
            p_dch = min(power, self.P_nom, p_dch_max_soe)
            p_ch = 0

        bat_internal_loss = (1 - sqrt_eff) * (p_ch + p_dch)
        inv_loss = ((1 - self.inv_eff_ch) * p_ch) + ((1 - self.inv_eff_dch) * p_dch)

        delta_soe = (
            ((p_ch - p_dch) - inv_loss - bat_internal_loss) * self.dt / self.E_nom
        )
        next_soe = prev_soe + delta_soe

        return p_ch, p_dch, next_soe
