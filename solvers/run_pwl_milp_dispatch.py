from benchopt import BaseSolver
import pyomo.environ as pyo
import numpy as np

# Number of PWL segments over the operating range [min_op, P_nom].
# Overridden at runtime by the value read from ESS_CONFIG.json → pwl_efficiency_fits.n_pieces.
# Change that config value to adjust approximation resolution without touching this file.
_DEFAULT_N_PWL_PIECES = 10


class PieceWiseLinearMILPSolver(BaseSolver):

    name = "PWL MILP solver"
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
        inv_ch_coeffs,
        inv_dch_coeffs,
        batt_coeffs,
        n_pieces=None,
        **kwargs,
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

        self.soe_start = soe_start
        self.temp_start = temp_start

        self.inv_ch_coeffs = inv_ch_coeffs
        self.inv_dch_coeffs = inv_dch_coeffs
        self.batt_coeffs = batt_coeffs
        self.n_pieces = n_pieces if n_pieces is not None else _DEFAULT_N_PWL_PIECES

    def run(self, n_iter=1):
        model = self.setup_problem()
        bp = {b: self._breakpoints(b) for b in self.N_bess}
        self._greedy_warm_start(model, bp)
        solver = pyo.SolverFactory("appsi_highs")
        solver.options["mip_rel_gap"] = 0.01
        results = solver.solve(model, tee=True)
        if hasattr(results, "termination_condition"):
            tc = str(results.termination_condition).lower()
        else:
            tc = str(results.solver.termination_condition).lower()
        if "optimal" not in tc and "feasible" not in tc:
            raise RuntimeError(
                f"PWL MILP solver did not find a feasible solution (status: {tc})"
            )

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

    # ------------------------------------------------------------------
    # Warm-start heuristic
    # ------------------------------------------------------------------

    def _greedy_warm_start(self, model, bp):
        """Price-threshold greedy: charge cheapest 33%, discharge most expensive 33%."""
        n_bp = self.n_pieces + 2
        price = np.array(self.price)
        low = np.percentile(price, 33)
        high = np.percentile(price, 67)

        for b in self.N_bess:
            soe = self.soe_start[b]
            for t in self.T:
                if price[t] <= low and soe < 0.9:
                    p_ch, p_dch, b_op = self.P_nom[b], 0.0, 1
                elif price[t] >= high and soe > 0.1:
                    p_ch, p_dch, b_op = 0.0, self.P_nom[b], 0
                else:
                    p_ch, p_dch, b_op = 0.0, 0.0, 1

                loss_ch = (
                    max(
                        0.0,
                        self._eval_quadratic(self.inv_ch_coeffs, p_ch)
                        + self._eval_quadratic(self.batt_coeffs, p_ch),
                    )
                    if p_ch > 0
                    else 0.0
                )
                loss_dch = (
                    max(
                        0.0,
                        self._eval_quadratic(self.inv_dch_coeffs, p_dch)
                        + self._eval_quadratic(self.batt_coeffs, p_dch),
                    )
                    if p_dch > 0
                    else 0.0
                )
                soe = float(
                    np.clip(
                        soe
                        + (p_ch - p_dch - loss_ch - loss_dch) * self.dt / self.E_nom[b],
                        0.0,
                        1.0,
                    )
                )

                model.p_ch[b, t].set_value(p_ch)
                model.p_dch[b, t].set_value(p_dch)
                model.b_op[b, t].set_value(b_op)
                model.soe[b, t].set_value(soe)
                self._init_pwl(model.lam_ch, model.seg_ch, b, t, p_ch, bp[b], n_bp)
                self._init_pwl(model.lam_dch, model.seg_dch, b, t, p_dch, bp[b], n_bp)

        for t in self.T:
            net = (
                self.load[t]
                - self.renewable_generation[t]
                - sum(
                    model.p_dch[b, t].value - model.p_ch[b, t].value
                    for b in self.N_bess
                )
            )
            if net > 0:
                model.p_grid_purchase[t].set_value(net)
                model.p_grid_sale[t].set_value(0.0)
                model.b_grid[t].set_value(0)
            else:
                model.p_grid_purchase[t].set_value(0.0)
                model.p_grid_sale[t].set_value(min(-net, self.renewable_generation[t]))
                model.b_grid[t].set_value(1)

    def _init_pwl(self, lam_var, seg_var, b, t, p, bps, n_bp):
        """Set lambda and segment initial values for a given power level p."""
        n_seg = n_bp - 1
        if p == 0.0:
            for k in range(n_bp):
                lam_var[b, t, k].set_value(1.0 if k == 0 else 0.0)
            for s in range(n_seg):
                seg_var[b, t, s].set_value(1 if s == 0 else 0)
            return
        seg_idx = n_seg - 1
        for s in range(n_seg):
            if bps[s] <= p <= bps[s + 1]:
                seg_idx = s
                break
        span = bps[seg_idx + 1] - bps[seg_idx]
        alpha = (p - bps[seg_idx]) / span if span > 0 else 0.0
        for k in range(n_bp):
            lam_var[b, t, k].set_value(
                (1.0 - alpha) if k == seg_idx else (alpha if k == seg_idx + 1 else 0.0)
            )
        for s in range(n_seg):
            seg_var[b, t, s].set_value(1 if s == seg_idx else 0)

    # ------------------------------------------------------------------
    # PWL helpers
    # ------------------------------------------------------------------

    def _eval_quadratic(self, coeffs, p_W):
        """Evaluate quadratic loss polynomial. Input p in W, output in W."""
        p_kW = p_W / 1000.0
        return coeffs[0] * p_kW**2 + coeffs[1] * p_kW + coeffs[2]

    def _breakpoints(self, b):
        """
        Breakpoints in W for battery b.
        Index 0 is the idle point (p=0). Indices 1..n_pieces+1 span [min_op, P_nom]
        evenly, giving n_pieces segments over the operating range.
        """
        return [0.0] + list(
            np.linspace(self.min_operation_power[b], self.P_nom[b], self.n_pieces + 1)
        )

    def _loss_breakpoints(self, b, direction):
        """
        Total loss [W] at each breakpoint for battery b.

        At p=0 (idle): losses are forced to 0 regardless of what the quadratic fit
        produces there, since the fits are only valid over the operating range.

        Negative loss values from the fit (possible at low power for the battery
        internal fit) are clipped to 0.
        """
        bps = self._breakpoints(b)
        losses = [0.0]  # k=0: idle point
        inv_coeffs = self.inv_ch_coeffs if direction == "ch" else self.inv_dch_coeffs
        for p in bps[1:]:
            inv_loss = max(0.0, self._eval_quadratic(inv_coeffs, p))
            batt_loss = max(0.0, self._eval_quadratic(self.batt_coeffs, p))
            losses.append(inv_loss + batt_loss)
        return losses

    # ------------------------------------------------------------------
    # Problem formulation
    # ------------------------------------------------------------------

    def setup_problem(self) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()

        model.T = pyo.Set(initialize=self.T)
        model.B = pyo.Set(initialize=self.N_bess)

        # K: breakpoint indices — 0 is idle (p=0), 1..n_pieces+1 is operating range
        n_bp = self.n_pieces + 2
        model.K = pyo.RangeSet(0, n_bp - 1)

        # Precompute per-battery breakpoints and losses (pure Python, outside Pyomo)
        bp = {b: self._breakpoints(b) for b in self.N_bess}
        loss_ch_bp = {b: self._loss_breakpoints(b, "ch") for b in self.N_bess}
        loss_dch_bp = {b: self._loss_breakpoints(b, "dch") for b in self.N_bess}

        # --- BESS power variables [W] ---
        def power_bounds(m, b, t):
            return 0, self.P_nom[b]

        model.p_ch = pyo.Var(model.B, model.T, bounds=power_bounds)
        model.p_dch = pyo.Var(model.B, model.T, bounds=power_bounds)

        # --- State of Energy (normalised 0–1) ---
        model.soe = pyo.Var(model.B, model.T, bounds=(0, 1))

        # --- Binary direction variable: 1 = charging, 0 = discharging ---
        model.b_op = pyo.Var(model.B, model.T, within=pyo.Binary)

        # S: segment indices (one fewer than breakpoints)
        model.S = pyo.RangeSet(0, n_bp - 2)

        # --- PWL convex-combination (lambda) variables ---
        model.lam_ch = pyo.Var(model.B, model.T, model.K, bounds=(0, 1))
        model.lam_dch = pyo.Var(model.B, model.T, model.K, bounds=(0, 1))

        # Binary segment selectors: seg=1 means that segment is the active PWL segment.
        # These implement SOS2 without relying on Pyomo's SOSConstraint API.
        model.seg_ch = pyo.Var(model.B, model.T, model.S, within=pyo.Binary)
        model.seg_dch = pyo.Var(model.B, model.T, model.S, within=pyo.Binary)

        # Convexity: lambdas must sum to 1 for each (b, t)
        def lam_ch_sum(m, b, t):
            return sum(m.lam_ch[b, t, k] for k in m.K) == 1

        model.lam_ch_sum = pyo.Constraint(model.B, model.T, rule=lam_ch_sum)

        def lam_dch_sum(m, b, t):
            return sum(m.lam_dch[b, t, k] for k in m.K) == 1

        model.lam_dch_sum = pyo.Constraint(model.B, model.T, rule=lam_dch_sum)

        # Exactly one segment active per (b, t)
        def seg_ch_sum(m, b, t):
            return sum(m.seg_ch[b, t, s] for s in m.S) == 1

        model.seg_ch_sum = pyo.Constraint(model.B, model.T, rule=seg_ch_sum)

        def seg_dch_sum(m, b, t):
            return sum(m.seg_dch[b, t, s] for s in m.S) == 1

        model.seg_dch_sum = pyo.Constraint(model.B, model.T, rule=seg_dch_sum)

        # SOS2 via segment selectors: lambda[k] can only be nonzero at the endpoints
        # of the active segment. For breakpoint k: lam[k] <= seg[k-1] + seg[k],
        # with seg[-1] = seg[n_bp-1] = 0 (boundary clamps).
        def sos2_ch(m, b, t, k):
            left = m.seg_ch[b, t, k - 1] if k > 0 else 0
            right = m.seg_ch[b, t, k] if k < n_bp - 1 else 0
            return m.lam_ch[b, t, k] <= left + right

        model.sos2_ch = pyo.Constraint(model.B, model.T, model.K, rule=sos2_ch)

        def sos2_dch(m, b, t, k):
            left = m.seg_dch[b, t, k - 1] if k > 0 else 0
            right = m.seg_dch[b, t, k] if k < n_bp - 1 else 0
            return m.lam_dch[b, t, k] <= left + right

        model.sos2_dch = pyo.Constraint(model.B, model.T, model.K, rule=sos2_dch)

        # PWL power linking: p_ch / p_dch are fully determined by their lambda weights
        def pwl_p_ch(m, b, t):
            return m.p_ch[b, t] == sum(m.lam_ch[b, t, k] * bp[b][k] for k in m.K)

        model.pwl_p_ch = pyo.Constraint(model.B, model.T, rule=pwl_p_ch)

        def pwl_p_dch(m, b, t):
            return m.p_dch[b, t] == sum(m.lam_dch[b, t, k] * bp[b][k] for k in m.K)

        model.pwl_p_dch = pyo.Constraint(model.B, model.T, rule=pwl_p_dch)

        # Prevent simultaneous charge/discharge
        def charge_limit(m, b, t):
            return m.p_ch[b, t] <= self.P_nom[b] * m.b_op[b, t]

        model.charge_limit = pyo.Constraint(model.B, model.T, rule=charge_limit)

        def min_charge_power(m, b, t):
            return m.p_ch[b, t] >= self.min_operation_power[b] * m.b_op[b, t]

        model.min_charge_power = pyo.Constraint(model.B, model.T, rule=min_charge_power)

        def discharge_limit(m, b, t):
            return m.p_dch[b, t] <= self.P_nom[b] * (1 - m.b_op[b, t])

        model.discharge_limit = pyo.Constraint(model.B, model.T, rule=discharge_limit)

        def min_discharge_power(m, b, t):
            return m.p_dch[b, t] >= self.min_operation_power[b] * (1 - m.b_op[b, t])

        model.min_discharge_power = pyo.Constraint(
            model.B, model.T, rule=min_discharge_power
        )

        # SoE dynamics with PWL losses
        # delta_soe = (p_ch − p_dch − loss_ch(p_ch) − loss_dch(p_dch)) * dt / E_nom
        # When b_op=0: p_ch=0 → lam_ch[k=0]=1 → loss_ch=0.  Same logic for discharging.
        def soe_balance(m, b, t):
            prev = self.soe_start[b] if t == 0 else m.soe[b, t - 1]

            loss_ch = sum(m.lam_ch[b, t, k] * loss_ch_bp[b][k] for k in m.K)
            loss_dch = sum(m.lam_dch[b, t, k] * loss_dch_bp[b][k] for k in m.K)

            delta_soe = (
                (m.p_ch[b, t] - m.p_dch[b, t] - loss_ch - loss_dch)
                * self.dt
                / self.E_nom[b]
            )
            return m.soe[b, t] == prev + delta_soe

        model.soe_balance = pyo.Constraint(model.B, model.T, rule=soe_balance)

        # --- Grid variables ---
        p_grid_max = 1e6
        model.p_grid_sale = pyo.Var(model.T, bounds=(0, p_grid_max))
        model.p_grid_purchase = pyo.Var(model.T, bounds=(0, p_grid_max))

        # Binary: 1 = feed-in, 0 = consume
        model.b_grid = pyo.Var(model.T, within=pyo.Binary)

        def grid_feed_in_rule(m, t):
            return m.p_grid_sale[t] <= p_grid_max * m.b_grid[t]

        model.grid_feed_in_rule = pyo.Constraint(model.T, rule=grid_feed_in_rule)

        def grid_feed_in_bound_to_renewable(m, t):
            return m.p_grid_sale[t] <= self.renewable_generation[t]

        model.grid_feed_in_bound_to_renewable = pyo.Constraint(
            model.T, rule=grid_feed_in_bound_to_renewable
        )

        def grid_purchase_rule(m, t):
            return m.p_grid_purchase[t] <= p_grid_max * (1 - m.b_grid[t])

        model.grid_purchase_rule = pyo.Constraint(model.T, rule=grid_purchase_rule)

        def power_flow_rule(m, t):
            return self.load[t] - self.renewable_generation[t] == (
                m.p_grid_purchase[t] - m.p_grid_sale[t]
            ) + sum(m.p_dch[b, t] - m.p_ch[b, t] for b in m.B)

        model.power_flow_rule = pyo.Constraint(model.T, rule=power_flow_rule)

        # Objective: minimise net energy cost
        def obj(m):
            return sum(
                (
                    m.p_grid_purchase[t] * self.price[t]
                    - m.p_grid_sale[t] * self.renewable_feed_in_tariff
                )
                * self.dt
                for t in m.T
            )

        model.obj = pyo.Objective(rule=obj, sense=pyo.minimize)
        return model
