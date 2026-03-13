import numpy as np
from benchopt import BaseSolver


class BaselineSolver(BaseSolver):

    name = "Baseline solver"

    parameters = {}

    def set_objective(
        self,
        T,
        N_bess,
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
    ):

        self.T = range(T)
        self.B = range(N_bess)
        self.dt = dt

        self.load = load
        self.renewable_generation = renewable_generation
        self.price = price
        self.renewable_feed_in_tariff = renewable_feed_in_tariff

        self.P_nom = P_nom
        self.min_operation_power = min_operation_power
        self.E_nom = E_nom
        self.batt_eff = batt_eff
        self.inv_eff_ch = inv_eff_ch
        self.inv_eff_dch = inv_eff_dch

        self.soe_start = soe_start
        self.temp_start = temp_start

    def run(self, n_iter=1):
        self.p_grid_sale = np.zeros_like(self.T)
        self.p_grid_purchase = np.zeros_like(self.T)
        self.p_ch = np.zeros((len(self.B), len(self.T)))
        self.p_dch = np.zeros((len(self.B), len(self.T)))
        self.soe = np.zeros((len(self.B), len(self.T)))

        for t in self.T:

            surplus = self.load[t] - self.renewable_generation[t]
            if t == 0:
                prev_soe = self.soe_start
            else:
                prev_soe = self.soe[:, t - 1]

            for b in self.B:
                self.p_ch[b, t], self.p_dch[b, t], self.soe[b, t], surplus = (
                    self.apply_bess_power(prev_soe=prev_soe[b], power=surplus, b=b)
                )

            self.p_grid_sale[t], self.p_grid_purchase[t] = self.apply_grid_power(
                surplus=self.load[t] - self.renewable_generation[t],
                p_ch=sum(self.p_ch[b, t] for b in self.B),
                p_dch=sum(self.p_dch[b, t] for b in self.B),
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

    def apply_bess_power(self, prev_soe, power, b):
        sqrt_eff = np.sqrt(self.batt_eff[b])
        alpha_ch = 1 - (1 - self.inv_eff_ch[b]) - (1 - sqrt_eff)
        alpha_dch = 1 + (1 - self.inv_eff_dch[b]) + (1 - sqrt_eff)

        # max allowed by SoE
        p_ch_max_soe = (1 - prev_soe) * self.E_nom[b] / (alpha_ch * self.dt)
        p_dch_max_soe = prev_soe * self.E_nom[b] / (alpha_dch * self.dt)

        power = np.clip(power, -self.P_nom[b], self.P_nom[b])
        if power < 0:  # charging
            p_ch = min(-power, self.P_nom[b], p_ch_max_soe)
            p_dch = 0

        else:  # discharging
            p_dch = min(power, self.P_nom[b], p_dch_max_soe)
            p_ch = 0

        if abs(p_ch) < self.min_operation_power[b]:
            p_ch = 0
        if abs(p_dch) < self.min_operation_power[b]:
            p_dch = 0

        bat_internal_loss = (1 - sqrt_eff) * (p_ch + p_dch)
        inv_loss = ((1 - self.inv_eff_ch[b]) * p_ch) + (
            (1 - self.inv_eff_dch[b]) * p_dch
        )

        delta_soe = (
            ((p_ch - p_dch) - inv_loss - bat_internal_loss) * self.dt / self.E_nom[b]
        )
        next_soe = prev_soe + delta_soe
        rest_surplus = power - (p_dch - p_ch)
        return p_ch, p_dch, next_soe, rest_surplus
