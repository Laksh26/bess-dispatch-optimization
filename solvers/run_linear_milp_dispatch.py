from benchopt import BaseSolver
import pyomo.environ as pyo
import numpy as np


class LinearMILPSolver(BaseSolver):

    name = "Pyomo based MILP solver"

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
        self.N_bess = range(N_bess)
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

        model = self.setup_problem()
        solver = pyo.SolverFactory("highs")
        results = solver.solve(model, tee=False)
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError("MILP solver did not converge to optimal solution")

        self.p_grid_sale = np.array(
            [float(pyo.value(model.p_grid_sale[t])) for t in model.T]
        )
        self.p_grid_purchase = np.array(
            [float(pyo.value(model.p_grid_purchase[t])) for t in model.T]
        )
        self.p_ch = np.array(
            [[float(pyo.value(model.p_ch[b, t])) for t in model.T] for b in model.B]
        )
        self.p_dch = np.array(
            [[float(pyo.value(model.p_dch[b, t])) for t in model.T] for b in model.B]
        )
        self.soe = np.array(
            [[float(pyo.value(model.soe[b, t])) for t in model.T] for b in model.B]
        )

    def get_result(self):
        return dict(
            p_grid_sale=self.p_grid_sale,
            p_grid_purchase=self.p_grid_purchase,
            p_ch=self.p_ch,
            p_dch=self.p_dch,
            soe=self.soe,
        )

    def setup_problem(self) -> pyo.ConcreteModel:
        """
        Models the dynamic problem with constraints and objectives. This could be further extended to make the problem
        more complex with various other constraints and variables.

        Implemented constraints:
        - SoE calculation of the BESS
        - Restricting grid power sales with renewable generation, because if sold more it is not paid
        - power flow constraint
        -
        :return:
        """
        model = pyo.ConcreteModel()

        model.T = pyo.Set(initialize=self.T)
        model.B = pyo.Set(initialize=self.N_bess)

        # power of BESS inverter - ch: Charging direction, dch: Discharging direction
        def power_bounds(m, b, t):
            return 0, self.P_nom[b]

        model.p_ch = pyo.Var(model.B, model.T, bounds=power_bounds)
        model.p_dch = pyo.Var(model.B, model.T, bounds=power_bounds)

        # modeled State of Energy of the BESS
        def soe_bounds(m, b, t):
            return 0, 1

        model.soe = pyo.Var(model.B, model.T, bounds=soe_bounds)

        # Binary variable to let the BESS to either charge or discharge at a given T. 1 --> charging, 0--> Discharging
        # When power is 0, b_op is valid with both 1 or 0
        model.b_op = pyo.Var(model.B, model.T, within=pyo.Binary)

        # prevent simultaneous charge/discharge
        def charge_limit(m, b, t):
            return m.p_ch[b, t] <= self.P_nom[b] * m.b_op[b, t]

        model.charge_limit = pyo.Constraint(model.B, model.T, rule=charge_limit)

        # min charge power
        def min_charge_power(m, b, t):
            return m.p_ch[b, t] >= self.min_operation_power[b] * m.b_op[b, t]

        model.min_charge_power = pyo.Constraint(model.B, model.T, rule=min_charge_power)

        def discharge_limit(m, b, t):
            return m.p_dch[b, t] <= self.P_nom[b] * (1 - m.b_op[b, t])

        model.discharge_limit = pyo.Constraint(model.B, model.T, rule=discharge_limit)

        # min discharge power
        def min_discharge_power(m, b, t):
            return m.p_dch[b, t] >= self.min_operation_power[b] * (1 - m.b_op[b, t])

        model.min_discharge_power = pyo.Constraint(
            model.B, model.T, rule=min_discharge_power
        )

        # SoE dynamics
        def soe_rule(m, b, t):

            if t == 0:
                prev = self.soe_start[b]
            else:
                prev = m.soe[b, t - 1]

            # calculated per direction
            bat_internal_loss = (1 - np.sqrt(self.batt_eff[b])) * (
                m.p_ch[b, t] + m.p_dch[b, t]
            )

            # inverter losses
            inv_loss = ((1 - self.inv_eff_ch[b]) * m.p_ch[b, t]) + (
                (1 - self.inv_eff_dch[b]) * m.p_dch[b, t]
            )

            delta_soe = (
                ((m.p_ch[b, t] - m.p_dch[b, t]) - inv_loss - bat_internal_loss)
                * self.dt
                / self.E_nom[b]
            )

            return m.soe[b, t] == prev + delta_soe

        model.soe_balance = pyo.Constraint(model.B, model.T, rule=soe_rule)

        # grid power
        p_grid_max = 1e6
        model.p_grid_sale = pyo.Var(model.T, bounds=(0, p_grid_max))
        model.p_grid_purchase = pyo.Var(model.T, bounds=(0, p_grid_max))

        # Binary variable to let the grid to either feed in or consume at a given T. 1 --> feed-in, 0--> consume
        model.b_grid = pyo.Var(model.T, within=pyo.Binary)

        def grid_feed_in_rule(m, t):
            return m.p_grid_sale[t] <= p_grid_max * m.b_grid[t]

        model.grid_feed_in_rule = pyo.Constraint(model.T, rule=grid_feed_in_rule)

        def grid_feed_in_bound_to_renewable_generation_rule(m, t):
            return m.p_grid_sale[t] <= self.renewable_generation[t]

        model.grid_feed_in_bound_to_renewable_generation_rule = pyo.Constraint(
            model.T, rule=grid_feed_in_bound_to_renewable_generation_rule
        )

        def grid_consume_rule(m, t):
            return m.p_grid_purchase[t] <= p_grid_max * (1 - m.b_grid[t])

        model.grid_purchase_rule = pyo.Constraint(model.T, rule=grid_consume_rule)

        def power_flow_rule(m, t):
            return self.load[t] - self.renewable_generation[t] == (
                m.p_grid_purchase[t] - m.p_grid_sale[t]
            ) + sum(m.p_dch[b, t] - m.p_ch[b, t] for b in m.B)

        model.power_flow_rule = pyo.Constraint(model.T, rule=power_flow_rule)

        # objective
        def obj(m):

            return sum(
                (
                    (
                        (m.p_grid_purchase[t] * self.price[t])
                        - (m.p_grid_sale[t] * self.renewable_feed_in_tariff)
                    )
                    * self.dt
                )
                for t in m.T
            )

        model.obj = pyo.Objective(rule=obj, sense=pyo.minimize)
        return model
