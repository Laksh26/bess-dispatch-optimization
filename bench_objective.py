from benchopt import BaseObjective
import numpy as np


class Objective(BaseObjective):
    def get_one_result(self):
        return dict(
            p_grid_sale=np.zeros(self.T),
            p_grid_purchase=np.zeros(self.T),
            p_ch=np.zeros(self.T),
            p_dch=np.zeros(self.T),
            soe=np.zeros(self.T),
        )

    name = "BESS dispatch objective"
    parameters = {}

    def set_data(
        self,
        T,
        load,
        renewable_generation,
        price,
        renewable_feed_in_tariff,
        P_nom,
        E_nom,
        batt_eff,
        inv_eff_ch,
        inv_eff_dch,
        dt,
        soe_start,
    ):
        self.T = T
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

    def evaluate_result(self, p_grid_sale, p_grid_purchase, p_ch, p_dch, soe):
        """
        This objective is common among all the algorithms, but the solvers will have its own set of objective,
        which can be adapted as per its own models available
        """
        # capping the grid sale according to the generation as feed-in-tariff is provided only for the amount of
        # generation. if more energy is supplied, it is not paid
        p_grid_sale_capped = np.minimum(p_grid_sale, self.renewable_generation)

        feed_in_revenue = p_grid_sale_capped * self.renewable_feed_in_tariff * self.dt
        purchase_cost = p_grid_purchase * self.price * self.dt
        revenue = np.sum(purchase_cost) - np.sum(feed_in_revenue)

        return dict(value=revenue)

    def get_objective(self):
        return dict(
            T=self.T,
            load=self.load,
            renewable_generation=self.renewable_generation,
            price=self.price,
            renewable_feed_in_tariff=self.renewable_feed_in_tariff,
            P_nom=self.P_nom,
            E_nom=self.E_nom,
            batt_eff=self.batt_eff,
            inv_eff_ch=self.inv_eff_ch,
            inv_eff_dch=self.inv_eff_dch,
            dt=self.dt,
            soe_start=self.soe_start,
        )
