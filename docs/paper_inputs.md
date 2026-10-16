# Paper inputs — Methods and Results facts

Collected 2026-10-15 for the *International Journal of Hydrogen Energy* manuscript. Every number is copied from the repo,
with its source as `file:line` (line numbers as of commit f33444f plus the uncommitted Step 8a files). "NOT FOUND"
means the item is not in the repo. Nothing here is a new computation, except where it is marked "computed here".
Lopes (2010) PSA performance values are the author's simulations, not measurements (CLAUDE.md, "Measured vs simulated").

## 1. Model equations as implemented

Source: `src/h2psa/column.py` (C), `src/h2psa/cycle.py` (Y), `src/h2psa/isotherms.py` (I), `src/h2psa/kinetics.py` (K), `src/h2psa/transport.py` (T), `src/h2psa/design.py` (D). Line numbers refer to the working tree at commit `c68a7ad`.

Discretisation: 1D finite volumes, $N$ equal cells, $\Delta z = L/N$, cell centres $z_j=(j+\tfrac12)\Delta z$ (C:172-174); $N=30$ for the layered-bed design runs (D:22). Interior-face convective values are upwinded with a van Leer limiter, $\psi(r)=(r+|r|)/(1+|r|)$, $r$ smoothed with $\delta=10^{-10}$, first order next to the boundaries (C:31, C:101-114). Time integration: scipy BDF, finite-difference sparse Jacobian (C:415-425, C:442-443).

States per cell: $C_i$ ($i=1..n_s-1$), $\Delta P = P-P_\text{ref}(t)$, $\bar C_i$ (macropore), $\bar q_i$, $T_g$, $T_p$, $T_w$, plus cumulative mole counters (C:232-239). The last species (He in every model, D:98, Y:110, C:525) is not a state: its concentration comes from the ideal-gas law (C:283).

### 1.1 Component mass balance (gas phase) and continuity

Bulk gas, per cell (C:355-359):
$$
\varepsilon\,\frac{\partial C_i}{\partial t} = -\frac{\partial F_i}{\partial z} - (1-\varepsilon)\,a_p k_f\,(C_i - C_{s,i})
$$
Face flux, interior faces (C:315-322):
$$
F_i\big|_{j+1/2} = u_{j+1/2}\,C_i^{\text{vL}}\big|_{j+1/2} \;-\; \big(\varepsilon D_{ax}\big)_{j+1/2}\,\frac{C_{T,j}+C_{T,j+1}}{2}\,\frac{y_{i,j+1}-y_{i,j}}{\Delta z}
$$
- $u$ is the superficial velocity; dispersion acts on the mole-fraction gradient times the arithmetic-mean $C_T$ (C:320).
- $(\varepsilon D_{ax})_{j+1/2}$: harmonic mean of the cell values $\varepsilon_j D_{ax,j}$ (C:190-191).
- $C_{s,i}$: pellet-surface concentration, film resistance through a Biot number (C:200, C:355):
$$
C_{s,i}=\frac{\mathrm{Bi}_i\,C_i+\bar C_i}{1+\mathrm{Bi}_i},\qquad \mathrm{Bi}_i=\frac{a_p k_f R_p^2}{\varepsilon_p\,\Omega_m\,D_{p,i}}
$$
- Film coefficient per bed volume: `kfa` $=(1-\varepsilon)a_pk_f$ (C:201); $a_p = 2.0/R_p$ (`particle_specific_area` = 2.0 in units of $1/R_p$, `compact_column_lopes2010.yaml`, used at C:498, C:563).

Total concentration and pressure (ideal gas; C:283):
$$
C_{n_s} = \frac{P_\text{ref}(t)+\Delta P}{R\,T_g} - \sum_{i<n_s} C_i,\qquad C_T=\sum_i C_i = \frac{P}{RT_g}
$$
Pressure state equation (total mass + ideal gas, $P=C_TRT_g$; C:380):
$$
\frac{d\,\Delta P}{dt} = R\Big(T_g\sum_i \frac{\partial C_i}{\partial t} + C_T\,\frac{\partial T_g}{\partial t}\Big) - \frac{dP_\text{ref}}{dt}
$$
($\partial C_i/\partial t$ for all $n_s$ species, the last included, from the flux balance above.) $P_\text{ref}(t)$ is the pressure imposed at the one pressure-controlled end (C:257-262); on a BC change the state is rebased, $\Delta P \mathrel{+}= P_\text{ref,old}(t_\text{old}) - P_\text{ref,new}(0)$ (C:264-270, Y:213-216). Rationale (code comment C:232-235): carrying $P$ as a deviation keeps the integration error below the Ergun pressure drop.

Velocity: from the Ergun equation (1.4) at every face; no separate continuity equation for $u$.

Mixture properties (C:210-214, C:315): $M_i$ [kg mol⁻¹], $C_{p,i}$, $\mu_i$ interpolated from the NIST table **at the feed temperature only** (constant in time and space); $C_{v,i}=C_{p,i}-R$; $\mu=\sum_i y_i\mu_i$ (linear mole-fraction mixing); $\rho=\sum_i M_iC_i$.

### 1.2 Particle kinetics: bi-LDF (macropore + micropore)

Macropore LDF (C:199, C:358):
$$
\frac{\partial \bar C_i}{\partial t} = k_{\text{mac},i}\,(C_{s,i}-\bar C_i) - \frac{\rho_p}{\varepsilon_p}\frac{\partial \bar q_i}{\partial t},\qquad k_{\text{mac},i}=\frac{\Omega_m D_{p,i}}{R_p^2}
$$
Micropore LDF with Arrhenius temperature dependence at the pellet temperature (C:357):
$$
\frac{\partial \bar q_i}{\partial t} = \Omega_c\,\frac{D_{c0,i}}{r_c^2}\,\exp\!\Big(-\frac{E_{a,i}}{R\,T_p}\Big)\,\big(q_i^*(\bar C,T_p)-\bar q_i\big)
$$
- $q^*$ is evaluated at the macropore partial pressures $\bar P_i=\bar C_i R T_p/10^5$ [bar] (C:298); below $P_\text{min}=10^{-10}$ bar continued linearly, $q^*(P)=q^*(\max(P,P_\text{min}))\,P/\max(P,P_\text{min})$ (C:32, C:300-302).
- $\Omega_m=8$, $\Omega_c=3$ (HY2SEPS-AC) / $15$ (HY2SEPS-ZEO) (`hy2seps_*_lopes2010.yaml` l.16-17, read at C:562).
- $(D_{c0}/r_c^2, E_a)$: Arrhenius refit $\ln(D_c/r_c^2)=\ln D_0 - E_a/(RT)$, least squares in $\ln$ vs $1/T$ on the measured $D_c/r_c^2$ (K:14-19, used at C:553-554, C:564-565; AC5-KS: C:483-485). $E_a$ in kJ mol⁻¹, ×10³ in the model (C:197).
- Species absent from a layer's isotherm (He): $D_{c0}=0 \Rightarrow \partial\bar q/\partial t=0$ (C:196).
- $D_{p,i}$: Bosanquet without Knudsen, $D_{p,i}=D_{m,i}/\tau$, $\tau=2.0$ (T:86-91, T:139); $D_{m,i}$ from Chapman–Enskog/Blanc mixing at the feed composition, $T_\text{feed}$, $P_\text{high}$ (T:63-83; C:532-540, C:582). $D_p$ and hence $k_\text{mac}$ and $\mathrm{Bi}$ are temperature- and pressure-independent during the run.

### 1.3 Energy balances (gas, solid, wall)

Gas (C:365-373):
$$
\varepsilon C_T\,(\bar C_{p}-R)\,\frac{\partial T_g}{\partial t} =
\frac{\partial}{\partial z}\Big(\lambda\frac{\partial T_g}{\partial z}\Big)
- \bar u\,C_T\bar C_p\,\frac{\partial T_g}{\partial z}
+ \varepsilon R T_g\sum_i\frac{\partial C_i}{\partial t}
- (1-\varepsilon)a_p h_f (T_g-T_p)
- \frac{4h_w}{d_{wi}}(T_g-T_w)
$$
- $\bar C_p=\sum_i y_iC_{p,i}$ (C:365).
- Convection: $\bar u_j=\tfrac12(u_{j-1/2}+u_{j+1/2})$ times the difference of van Leer face temperatures (C:366-368); at an end face the inflow temperature $T_\text{in}$ if gas enters there, else the end-cell $T_g$.
- Conduction: face $\lambda$ = harmonic mean of cell values (C:191); conductive flux set to **zero at both end faces** (C:369-370).
- No heat-of-adsorption term and no enthalpy carried by the film mass flux in the gas balance.

Solid (pellet) (C:371, C:374-376):
$$
(1-\varepsilon)\Big[\varepsilon_p\sum_i C_{v,i}\bar C_i + \rho_p\sum_i C_{v,i}\bar q_i + \rho_p C_{ps}\Big]\frac{\partial T_p}{\partial t}
= (1-\varepsilon)\varepsilon_p R T_p\sum_i\frac{\partial \bar C_i}{\partial t}
+ (1-\varepsilon)\rho_p\sum_i(-\Delta H_i)\frac{\partial \bar q_i}{\partial t}
+ (1-\varepsilon)a_p h_f (T_g-T_p)
$$
- Adsorbed phase heat capacity $=C_{v,i}$ of the gas (C:374).
- $(-\Delta H_i)$: constant, `minus_dH`×10³ of the isotherm's $K$ (C:198); 0 for non-adsorbing species. For AC5-KS (virial) this is the Henry-constant heat, loading dependence neglected (`compact_column_lopes2010.yaml` `isosteric_heat`).
- No axial conduction in the solid.

Wall (C:217-220, C:377-378):
$$
\rho_wC_{p,w}\frac{\partial T_w}{\partial t} = \alpha_w h_w (T_g-T_w) - \alpha_{wl} U (T_w-T_\infty)
$$
$$
\alpha_w=\frac{d_{wi}}{e\,(d_{wi}+e)},\qquad \alpha_{wl}=\frac{1}{(d_{wi}+e)\ln\!\big((d_{wi}+e)/d_{wi}\big)},\qquad \rho_wC_{p,w}=\frac{C_w}{\pi e (d_{wi}+e)}
$$
- $C_w=\rho_wC_{p,w}\pi e(d_{wi}+e)$ (C:505-510); $T_\infty = T_\text{amb}=T_\text{feed}$ (D:100, Y:107).
- No axial conduction in the wall.
- Layered design runs: $h_w$, $U$ taken from `ac5ks_lopes2010.yaml` `psa_transport.groups["1-4"]` for every layer (D:97, D:101, C:583-584), not from the correlations T:131/T:138.
- `isothermal=True` option sets $\partial T_g/\partial t=\partial T_p/\partial t=\partial T_w/\partial t=0$ (C:361-362).

### 1.4 Momentum (Ergun), quasi-steady

Interior faces (C:96-98, C:316-319):
$$
-\frac{\partial P}{\partial z} = a\,u + b\,|u|\,u,\qquad a=\frac{150\,\mu(1-\varepsilon)^2}{\varepsilon^3 d_p^2},\qquad b=\frac{1.75\,(1-\varepsilon)\rho}{\varepsilon^3 d_p}
$$
solved explicitly for $u$ (stable root):
$$
u_{j+1/2}=\frac{2G}{a_f+\sqrt{a_f^2+4b_f|G|}},\qquad G=-\frac{\Delta P_{j+1}-\Delta P_j}{\Delta z}
$$
- $a_f,b_f$: arithmetic means of the adjacent cell values (C:318); $d_p=2R_p$ (C:202); $\mu,\rho$ as in 1.1.
- Pressure-controlled end: same law over the half cell, with the **end-cell** (not face-averaged) $a,b$ (C:335-336):
  $u_\text{end}=\text{Ergun}\big(\text{drop}/(\Delta z/2)\big)$, $\text{drop}=P_\text{ref}-P_0$ at $z=0$, $P_{N-1}-P_\text{ref}$ at $z=L$.

### 1.5 Boundary conditions by end type and by step

End types (`BC`, C:80-93; implementation C:323-343). Exactly one end is pressure-controlled (C:257). $\mathrm{sgn}=+1$ at $z=0$, $-1$ at $z=L$.

| kind | mass flux at end face | velocity | inflow temperature |
|---|---|---|---|
| `closed` | $F_i=0$ | $u=0$ | none (end-cell $T_g$ in the upwind convective term) |
| `flow` (Danckwerts, total flux) | $F_i=\mathrm{sgn}\,\dot n/A\;y_{b,i}$ | $u=\mathrm{sgn}\,\dot n/(A\,C_{T,\text{end}})$ | $T_b$ |
| `pressure` | outflow, or inflow with no `y`: $F_i=u\,C_{i,\text{end}}$ ($\partial C/\partial z=0$); inflow with `y`/`y_t`: $F_i=u\,y_{b,i}\,P_\text{ref}/(R\,T_b)$ | half-cell Ergun (1.4) | $T_b$ if inflow and `T` given, else end-cell $T_g$ |

No dispersive flux is added at the end faces in any case. `y_t(t)` overrides `y` when set (C:333-334).

Steps (Y:57-62, Y:137-169). $P_\text{start}$ = cell-centre pressure of the last cell ($P_{N-1}$) for all steps except Blow and Purge, which use the first cell ($P_0$) (Y:211-212). All inflows at $T_\text{feed}$ (Y:18, Y:132, Y:157, Y:163-168).

| step | $z=0$ (feed end) | $z=L$ (product end) | pressure law / inflow |
|---|---|---|---|
| Feed | `flow`: $\dot n_\text{feed}$, $y_\text{feed}$, $T_\text{feed}$ | `pressure`: $P_\text{high}$ const. | outflow at $z=L$ (Y:132-133, Y:150-151) |
| ED$k$ | closed | `pressure`: exponential to $P_{\text{inter},k}$ | no `y`, no `T` (Y:152-153) |
| Blow | `pressure`: exponential to $P_\text{low}$ | closed | no `y`, no `T` (Y:154-155) |
| Purge | `pressure`: $P_\text{low}$ const. | `flow`: $\dot n_\text{purge}$ pure H₂, $T_\text{feed}$ | (Y:156-157) |
| EP$k$ | closed | `pressure`: exponential to $P_{\text{inter},k}$ | inflow per `ep_gas`, $T_\text{feed}$ (Y:158-165) |
| Press | closed | `pressure`: linear $P_\text{start}\to P_\text{high}$ | pure H₂, $T_\text{feed}$ (Y:166-168) |
| Hold / idle | NOT FOUND | | |

Order: Feed, ED1..ED$n$, Blow, Purge, EP$n$..EP1, Press (Y:60-62); ED$k$ and EP$k$ share $P_{\text{inter},k}$, $k=1$ the highest. Design runs: $P_{\text{inter},k}=P_\text{high}-k(P_\text{high}-P_\text{low})/(n+1)$ (D:89).

EP inflow composition `ep_gas` (Y:11-16, Y:158-165, Y:226-232):
- `"h2"`: pure H₂ — **frozen baseline** (`cycle_B_experiment.ep_inflow: h2`, read at Y:86).
- `"ed"`: mole fractions of the total (clipped ≥0) outflow of ED$k$ of the same cycle (Y:226-228); Step 7b scenario `ep_inflow_ed_gas` (D:62-63).
- `"product"`: well-mixed Feed-step product of the same cycle (Y:230-232).
- `"ed_direct"` / `"ed_reversed"`: time-dependent $y(t)$ from the ED$k$ outlet history binned in 20 equal-mole bins, mapped by $\phi=t/t_\text{eq}$ or $1-t/t_\text{eq}$ (Y:159-163, Y:171-180).

### 1.6 Pressure law ("valve law") with $\beta_P$

No valve equation (no $C_v$ flow law): the end pressure is imposed and the flow follows from half-cell Ergun (1.4). Exponential law for ED, Blow and EP, re-anchored at the step-start pressure (Y:6-8, Y:145-147):
$$
P_\text{end}(t) = P_\text{target} + (P_\text{start}-P_\text{target})\,e^{-\beta_P t},\qquad
\frac{dP_\text{end}}{dt} = -\beta_P\,(P_\text{start}-P_\text{target})\,e^{-\beta_P t}
$$
$t$ = time since step start. Press, linear (Y:166-168):
$$
P_\text{end}(t)=P_\text{start}+\frac{P_\text{high}-P_\text{start}}{t_\text{press}}\,t
$$
Feed and Purge: constant $P_\text{high}$, $P_\text{low}$ (Y:142-143).
- $\beta_P$: one constant for ED, Blow and EP (Y:17, Y:52-54), or per step type via `beta_steps` = {"ED","Blow","EP"} (Y:50).
- Baseline value: `fitted.pressure_decay_rate_author_sims` = 0.102 s⁻¹ (`beta_P_source`, read at Y:84; `compact_column_lopes2010.yaml` l.36, l.96). Alternatives in YAML: 0.1998 s⁻¹ (fit to Fig. 5.18a); per step ED 7.33, Blow 0.078, EP 0.087 s⁻¹.

### 1.7 Isotherms and layer interface

Layered bed (HY2SEPS-AC at the feed end, HY2SEPS-ZEO at the product end; C:568-587, D:93-101): multisite Langmuir (Nitta) (I:292-357):
$$
\frac{q_i^*}{q_{\max,i}} = a_i K_i P_i\Big(1-\sum_j \frac{q_j^*}{q_{\max,j}}\Big)^{a_i},\qquad K_i=K_{\infty,i}\exp\!\Big(\frac{-\Delta H_i}{RT}\Big)
$$
$P_i$ in kPa inside (bar ×100, I:345), $K_\infty$ in kPa⁻¹. Solved by Newton in $u=\ln s$, $s=1-\sum\theta$, with $s-1+\sum_i c_is^{a_i}=0$, $c_i=a_iK_iP_i$, $u$ capped at 0, $P$ clipped at 0 (I:337-357). Capacity disturbance $\phi$ multiplies every $q_{\max}$ ($a$, $K$ unchanged) (C:558-559).

Compact AC5-KS column (validation, Y:105-107, C:494-495): virial with Taqvi–LeVan mixing (I:65-87, I:244-288):
$$
P_i=\frac{q_i}{K_{H,i}}\exp\!\Big(\frac{2}{S}\sum_jA_{ij}q_j+\frac{3}{2S^2}\sum_{j,k}B_{ijk}q_jq_k\Big),\quad A_{ij}=\tfrac{A_i+A_j}{2},\; B_{ijk}=\tfrac{B_i+B_j+B_k}{3}
$$
$K_{H,i}=K_{\infty,i}e^{-\Delta H_i/RT}$, $A_i=A_{0,i}+A_{1,i}/T$, $B_i=B_{0,i}+B_{1,i}/T$ (I:55-62). Options `iast`, `noncompetitive_h2` (I:68-73, I:131-146).

Layer interface (C:177-207, C:190-191, C:318; D:94-96):
- Every interface must lie on a cell face (else `ValueError`; design checks $f_\text{zeolite}N\in\mathbb{Z}$).
- Each cell carries its own layer's $\varepsilon$, $\rho_p$, $\varepsilon_p$, $R_p$, $C_{ps}$, $a_p$, $k_f$, $h_f$, $\Omega_m$, $\Omega_c$, $D_p$, $D_{c0}$, $E_a$, $-\Delta H$, isotherm, and $D_{ax}$, $\lambda$, $h_w$, $U$ (layer value or column default).
- At the interface face: $\varepsilon D_{ax}$ and $\lambda$ harmonic mean; Ergun $a$, $b$ arithmetic mean; $C_i$, $T_g$ single continuous fields (upwinded as anywhere else). No explicit jump condition.
- Layer transport coefficients from correlations at feed conditions ($y_\text{feed}$, $T_\text{feed}$, $P_\text{high}$, $u_0=\dot n RT/(PA)$) (C:577, C:582; T:106-141): $D_{ax}=(0.45+0.55\varepsilon)D_m+0.35R_pu_0/\varepsilon$ (eq. 3.18, T:94-96, T:127); $\lambda=k_g(7+0.5\,\mathrm{Pr\,Re})$ (T:128); $h_f=\frac{k_g}{d_p}(2+1.1\mathrm{Re}^{0.6}\mathrm{Pr}^{1/3})$ (T:129); $k_f=\frac{D_m}{d_p}(2+1.1\mathrm{Re}^{0.6}\mathrm{Sc}^{1/3})$ (T:130); $D_m=\sum_i y_iD_{m,i}$ (T:122).

### 1.8 Symbol table

| Symbol | SI unit | Code name (file:line) |
|---|---|---|
| $C_i$ | mol m⁻³ | `C` (C:283) |
| $C_T$ | mol m⁻³ | `CT` (C:310) |
| $y_i$ | – | `yfr` (C:311) |
| $P$, $\Delta P$, $P_\text{ref}$ | Pa | `P`, `dP`, `P_ref` / `Pref` (C:262, C:282, C:313) |
| $\bar C_i$ | mol m⁻³ (pore fluid) | `Cm` (C:284) |
| $C_{s,i}$ | mol m⁻³ | `Cs` (C:355) |
| $\bar q_i$, $q^*_i$ | mol kg⁻¹ | `q`, `_qstar` (C:284, C:290) |
| $T_g,T_p,T_w$ | K | `Tg`, `Tp`, `Tw` (C:281, C:285) |
| $T_\infty$ | K | `col.T_amb` (C:68) |
| $u$ | m s⁻¹ (superficial) | `u`, `u_int`, `u_end` (C:319, C:344) |
| $F_i$ | mol m⁻² s⁻¹ | `F` (C:321) |
| $\varepsilon$ | – | `eps` (C:189) |
| $\varepsilon_p$ | – | `eps_p` (C:189) |
| $\rho_p$ | kg m⁻³ | `rho_p` (C:189) |
| $R_p$, $d_p$ | m | `Rp`, `dp` (C:192, C:202) |
| $a_p$ | m⁻¹ | `a_p` (C:192) |
| $k_f$ | m s⁻¹ | `kf` (C:193) |
| $h_f$ | W m⁻² K⁻¹ | `hf` (C:193) |
| $\mathrm{Bi}_i$ | – | `Bi` (C:200) |
| $\Omega_m$, $\Omega_c$ | – | `Om_m`, `Om_c` (C:194) |
| $D_{p,i}$ | m² s⁻¹ | `Dp` (C:195) |
| $k_{\text{mac},i}$ | s⁻¹ | `kmac` (C:199) |
| $D_{c0,i}/r_c^2$ | s⁻¹ | `Dc0` (C:196) |
| $E_{a,i}$ | J mol⁻¹ | `Ea` (C:197) |
| $-\Delta H_i$ | J mol⁻¹ | `dH` (C:198) |
| $D_{ax}$ | m² s⁻¹ | `Dax_c`; face $\varepsilon D_{ax}$: `eDax_f` (C:186, C:191) |
| $\lambda$ | W m⁻¹ K⁻¹ | `lam_c`, `lam_f` (C:186, C:191) |
| $h_w$ | W m⁻² K⁻¹ | `hw_c` (C:186) |
| $U$ | W m⁻² K⁻¹ | `U_c` (C:186) |
| $C_{ps}$ | J kg⁻¹ K⁻¹ | `Cps` (C:192) |
| $C_{p,i}$, $C_{v,i}$ | J mol⁻¹ K⁻¹ | `Cp`, `Cv` (C:212-213) |
| $\mu_i$, $\mu$ | Pa s | `mu` (C:214, C:315) |
| $M_i$ | kg mol⁻¹ | `M` (C:211) |
| $\rho$ | kg m⁻³ | `rho` (C:315) |
| $a$, $b$ (Ergun) | Pa s m⁻², Pa s² m⁻³ | `a`, `b`, `af`, `bf` (C:316-318) |
| $d_{wi}$ | m | `d_wi` (C:61) |
| $e$ | m | `e_wall` (C:62) |
| $C_w$ | J m⁻¹ K⁻¹ | `Cw` (C:63) |
| $\rho_wC_{p,w}$ | J m⁻³ K⁻¹ | `rhoCp_w` (C:220) |
| $\alpha_w$, $\alpha_{wl}$ | m⁻¹ | `alpha_w`, `alpha_wl` (C:218-219) |
| $A$ | m² | `A` (C:175) |
| $L$, $\Delta z$, $N$ | m, m, – | `L`, `dz`, `N` (C:172-173) |
| $\dot n$ (feed, purge) | mol s⁻¹ | `feed_flow`, `purge_flow` (Y:46, Y:48) |
| $\beta_P$ | s⁻¹ | `beta_P`, `beta_steps` (Y:47, Y:50) |
| $P_\text{high},P_\text{low},P_{\text{inter},k}$ | Pa | `P_high`, `P_low`, `P_inter` (Y:41-43) |
| $q_{\max,i}$ | mol kg⁻¹ | `MSLSpecies.q_max` (I:294) |
| $a_i$ (sites) | – | `MSLSpecies.a` (I:295) |
| $K_{\infty,i}$ (MSL) | kPa⁻¹ | `MSLSpecies.K_inf` (I:296) |
| $K_{\infty,i}$ (virial) | mol kg⁻¹ bar⁻¹ | `VirialSpecies.K_inf` (I:48) |
| $A_0,A_1,B_0,B_1$ | m² mol⁻¹, m² K mol⁻¹, m⁴ mol⁻², m⁴ K mol⁻² | `VirialSpecies.A0..B1` (I:50-53) |
| $S$ | (BET area, as in YAML) | `Virial.S` (I:79) |
| $\tau$ | – | `tau` (T:86) |
| $R$ | J mol⁻¹ K⁻¹ | `R = 8.314462618` (I:19) |

### Notes: discrepancies and non-standard choices

- `Cycle.ep_gas` defaults to `"ed"` (Y:49) and the module docstring lists `"ed"` first, but `experiment_cycle()` (used by every design run) takes `ep_inflow: h2` from the YAML (Y:86): the baseline EP inflow is pure H₂.
- The thesis exponential law (eqs. 5.39-5.40) is re-anchored to the step-start pressure, and the same law is applied to EP, for which the thesis gives no law (Y:6-8; YAML l.27-30).
- "Danckwerts" at a `flow` end (C:7, C:328) means the total molar flux equals the inflow (mass). For heat, the conductive flux at the end face is zero and $T_\text{in}$ enters only through the upwinded convective term (C:366-370).
- Gas properties ($C_p$, $\mu$, $M$) are fixed at $T_\text{feed}$ (C:209-214). Mixture viscosity in the column is linear in mole fraction (C:315); Wilke mixing is used only inside the transport correlations (T:50-53).
- Wall geometry uses $d_{wi}+e$ (C:218-219); the overall-$U$ correlation uses an outer diameter $d_{wi}+2e$ (T:101). Not used in design runs, where $U$ comes from the YAML (D:97-101).
- Momentum is quasi-steady Ergun (no accumulation term). The pressure-end half-cell uses cell, not face, $a,b$ (C:336).
- `kinetics.dc_over_rc2` (K:22-23) is not called by the column; the column evaluates the Arrhenius term inline (C:357).
- Hold step: NOT FOUND. Wall and solid axial conduction: NOT FOUND (not implemented).

# Paper-inputs fact sheet — sections 2 and 3

Repo: /Users/izbassar/Documents/Projects/h2-psa-digital-twin (HEAD c68a7ad). All `file:line` are relative to the repo root.
Page numbers are the thesis's printed pages, as quoted in the YAML (PDF page = printed + 34).

## 2. Numerics

### 2.1 Spatial discretisation

| item | value (as in repo) | source |
|---|---|---|
| method | "finite volumes; convection upwinded with the van Leer limiter" | src/h2psa/column.py:6 |
| limiter | van Leer, `(r + |r|) / (1 + |r|)`; smoothing `_DELTA = 1e-10` "(Haghpanah et al., 2013)" | src/h2psa/column.py:101-102, 31 |
| order at boundaries | "First order next to the boundaries" (limited upwind face values at the N-1 interior faces) | src/h2psa/column.py:105-114 |
| momentum | Ergun (5.4) evaluated at cell faces; end faces use Ergun through the half cell next to the end | src/h2psa/column.py:3, 7-8, 316-319, 335-336 |
| dispersion / conduction at faces | harmonic mean of the two cells (incl. at the layer interface) | src/h2psa/column.py:190-191; docs/validation/step4.md:68-71 |
| Jacobian sparsity | gas variables couple to cells j-2..j+2 (limiter stencil); everything else local | src/h2psa/column.py:388-413 |
| total cells (Step 5-8 runs) | `N_CELLS = 30` | src/h2psa/design.py:22, 79 |
| cells per layer | ZEO cells k = round(f_zeolite x N), AC cells N - k (`layered_column(1.0 - k / N, ...)`); f_zeolite must be a multiple of 1/30, else ValueError | src/h2psa/design.py:93-96, 100; configs/design_space.yaml:20-21 |
| uniform grid | dz = L / N; layer interfaces must lie on a cell face (checked) | src/h2psa/column.py:173, 177-181 |
| grid check (step 4) | N 30 -> 60, f_AC 0.6: recovery 0.007 pp (< 0.3 pp), impurity 7.9 % (< 10 %), PASS | docs/validation/step4.md:106-107 |
| original author's discretisation (context) | "gPROMS, orthogonal collocation on finite elements, 40 elements, 2 interior points, tol 1e-5" | data/params/compact_column_lopes2010.yaml:46 |
| model default N (not used in Step 6) | `ColumnModel(..., N: int = 60)`; `psa_model(..., N: int = 30)` | src/h2psa/column.py:168; src/h2psa/cycle.py:101 |

### 2.2 Time integration

| item | value | source |
|---|---|---|
| integrator | `scipy.integrate.solve_ivp(..., method="BDF", jac=self.jac, rtol=rtol, atol=self.atol)` | src/h2psa/column.py:442-443 |
| Jacobian | sparse forward-difference, column-coloured, step h = 1.5e-8 x max(|y|, yscale) | src/h2psa/column.py:415-425 |
| rtol | 1e-6 (`CycleRunner(..., rtol: float = 1e-6)`; `solve(..., rtol: float = 1e-6)`); run_batch.py does not override it | src/h2psa/cycle.py:123; src/h2psa/column.py:434; scripts/run_batch.py:185 |
| atol (vector) | gas C_i: 1e-6; pressure-deviation row dP: 1e-3 [Pa]; macropore C_m: 1e-6; loadings q: 1e-8; Tg, Tp, Tw: 1e-5; cumulative moles: 1e-9 | src/h2psa/column.py:242-247 |
| state layout | [G, Cm, q, Tg, Tp, Tw, n_in, n_out]; pressure carried as deviation dP = P - p_ref(t) | src/h2psa/column.py:232-236 |
| retry on BDF failure | same step re-solved with rtol / 10, then rtol x 10 (counted in `retries`) | src/h2psa/cycle.py:182-191 |
| max step | NOT FOUND (no `max_step` argument anywhere in src/ or scripts/) | src/h2psa/column.py:442-443 |
| formulation | ODE (method-of-lines) with the last species from the ideal-gas law; no DAE solver | src/h2psa/column.py:275-285 |
| isotherm solve (speed-up) | "multisite Langmuir solved by Newton in ln(s) with warm start (about 5x per cycle)" | docs/validation/step4.md:313-314 |

### 2.3 CSS criterion

| item | value | source |
|---|---|---|
| quantities | impurity ppm, H2 recovery, bed-averaged loadings `q_*` | src/h2psa/cycle.py:317, 277-278 |
| tolerance (relative change over window) | `tol: 1.0e-4` | configs/design_space.yaml:48; scripts/run_batch.py:204 |
| impurity tolerance (extra) | `tol_ppm: 1.0e-5` ("impurity change over the same cycles (step 4, acceleration check)") | configs/design_space.yaml:49; src/h2psa/cycle.py:323 |
| window | 10 cycles (default `window: int = 10`; not overridden by run_batch.py) | src/h2psa/cycle.py:273; scripts/run_batch.py:204-206 |
| mole balance | `abs(closure) < 1e-3` for every cycle in the window | src/h2psa/cycle.py:324-325 |
| minimum plain cycles | CSS declared only if `n_plain >= window` (>= 10 consecutive plain cycles) when acceleration is on | src/h2psa/cycle.py:324, 286-289; docs/preregistration.md:43-44 |
| cycle cap | `max_cycles: 2000` | configs/design_space.yaml:47; scripts/run_batch.py:110 |
| final cycle | one extra cycle after stop for traces/profiles; its perf is the reported label | src/h2psa/cycle.py:351-357 |
| cap handling | status "cap": last-cycle values used and flagged | docs/preregistration.md:59-60; scripts/run_batch.py:227 |

### 2.4 Warm start

| item | value | source |
|---|---|---|
| rule (Step 5-8 runner) | start from the CSS state of the nearest finished point of the same batch; distance = Euclidean norm in `design.scaled` ([0,1]^d over all decision + disturbance bounds) | scripts/run_batch.py:61-69, 168-171; src/h2psa/design.py:106-121 |
| rule (Step 4, earlier) | nearest case of same geometry/grid/adsorbents, Euclidean distance in f_AC, t_feed / 110 s, feed-flow ratio | docs/validation/step4.md:87-89 |
| first point of a batch | He-filled column ("starting empty") | scripts/run_batch.py:168-171; docs/preregistration.md:435-436 |
| only CSS states seed | `css_states/<id>.npz` written only when `res["css"]` | scripts/run_batch.py:234-235 |
| cold-start option | `--cold-start` (Step 7b spot check only) | scripts/run_batch.py:21, 303; docs/preregistration.md:383-385 |
| Step 7b | each point starts from its own baseline CSS state (distance 0) | docs/preregistration.md:378-381; scripts/run_step7b.py:63-76 |
| justification | "step-4 acceleration check bounds their effect on the labels (impurity within 1e-3 relative)" | docs/preregistration.md:77-78 |

### 2.5 Anderson acceleration and stall safeguard

| item | value | source |
|---|---|---|
| type | Anderson type II (Walker & Ni 2011) on the cycle map x = G(x), variables scaled by `yscale`; cumulative mole counters excluded | src/h2psa/cycle.py:375-380, 302-303 |
| when used | always in the batch runner (`accel=True`) | scripts/run_batch.py:204-206 |
| memory | 5 (`memory: int = 5`) | src/h2psa/cycle.py:274, 382; docs/validation/step4.md:89 |
| mixing / damping | NOT FOUND (no mixing parameter; undamped update `x_new = (gu - Gm @ gamma) * s`) | src/h2psa/cycle.py:406-408 |
| least squares | `np.linalg.lstsq(F, f, rcond=1e-10)` | src/h2psa/cycle.py:407 |
| non-negativity | gas and macropore concentrations, loadings clipped at 0 | src/h2psa/cycle.py:359-365, 409-410 |
| restart on growth | memory dropped if residual > `grow` (2.0) x best residual since restart | src/h2psa/cycle.py:382, 397-398 |
| switch to plain cycles | when max-norm scaled residual < `tol_acc` = 1e-6, or `stalled` | src/h2psa/cycle.py:274, 336-337 |
| stall safeguard | `stalled` when the residual has not halved in `stall` = 20 accelerated steps | src/h2psa/cycle.py:275, 396, 413-415; docs/validation/step4.md:91-93 |
| re-acceleration | after 3 x window (= 30) plain cycles without CSS, Anderson resumes with fresh memory | src/h2psa/cycle.py:343-344, 288-289 |
| verification | accelerated vs brute force: 14/14, ppm 25-1766, ppm within 1e-3 rel., recovery within 0.05 pp | docs/validation/step4.md:109 |

### 2.6 Censoring

| item | value | source |
|---|---|---|
| rule | stop when after >= `min_cycles` cycles the impurity exceeds `limit_ppm` and has increased over the last `window` cycles | configs/design_space.yaml:50-51; src/h2psa/cycle.py:368-372 |
| parameters | `min_cycles: 100, limit_ppm: 1000, window: 30` | configs/design_space.yaml:50 |
| exact condition | `len(ppm) >= max(min_cycles, window + 1) and ppm[-1] > limit and ppm[-1] > ppm[-1 - window]` (history includes accelerated iterates) | src/h2psa/cycle.py:371-372 |
| stored label | censored = last impurity X [ppm]; status "censored" | src/h2psa/cycle.py:329-332; scripts/run_batch.py:227 |
| response value | z = clip(log10(ppm), -1, 3); censored -> z = 3.0 | scripts/run_step6a.py:144-146; docs/preregistration.md:48-55; src/h2psa/step6.py:91-92 |
| objectives of censored points | excluded from recovery/productivity training | docs/preregistration.md:56-58; src/h2psa/step6.py:83-85 |
| history survives resume | `censor_history` kept in checkpoint | src/h2psa/cycle.py:293-294; scripts/run_batch.py:163, 177 |
| verification | 231 step-4 histories, 70 would stop, 0 below 1000 ppm at CSS; CSS impurity of stopped runs 1515-7.84e+04 ppm | docs/validation/step5.md:15, 57-58 |

### 2.7 Point timeout, retries, pause rules, execution

| item | value | source |
|---|---|---|
| point timeout | 2 h active wall time per attempt, summed over resumes (`--point-timeout 7200`); TimeoutError = solver failure | scripts/run_batch.py:18-20, 90-107, 203; docs/preregistration.md:327-332 |
| timeout scope | Step 6c (S_nom_0590 and all S_dist), Step 7b, Step 8a; NOT applied in Step 6a | scripts/run_step6c.py:26; scripts/run_step7b.py:40; scripts/run_step8a.py:54; scripts/run_step6a.py:266-267; docs/preregistration.md:332-334 |
| solver-failure retry | once, from the He-filled column, frozen tolerances (`--retry-failures`); failing again -> "failed", excluded from training | scripts/run_batch.py:15-17, 213-221; docs/preregistration.md:60-63 |
| pause thresholds | `--failure-limit 0.02`, `--cap-limit 0.05` (fractions of block size); pilot block size 32 | scripts/run_step6a.py:263-265; docs/preregistration.md:248-251 |
| pause action | stop new submissions, drain active points, exit 3; resuming needs "project-lead instruction" | scripts/run_batch.py:22-24, 350-361; scripts/run_step6a.py:305-306, 327-332 |
| worker (infrastructure) error | halts intake | scripts/run_batch.py:387-393 |
| Step 7b spot-check pause | any |z_cold - z_warm| > 0.01 (`TOL_DZ = 0.01`, 10 points/scenario, seed 7) or a failed run | scripts/run_step7b.py:39; docs/preregistration.md:382-386 |
| checkpoint | every 10 cycles; resume from it | scripts/run_batch.py:9-11, 196-200 |
| workers | 6 (deviation from the registered 8) | docs/preregistration.md:230-235, 310-314; scripts/run_step6a.py:363 (choices 6, 8) |
| threads per worker | OMP/OPENBLAS/MKL/VECLIB/NUMEXPR = 1 | scripts/run_step6a.py:270-272 |
| machine | Apple M1 Pro, 10 cores (8 performance + 2 efficiency), 16 GB | docs/validation/step5.md:77; docs/preregistration.md:223 |

### 2.8 Run time per point / per block

| set | value | source |
|---|---|---|
| Step 4 layered CSS, warm start + Anderson (31 runs, 7 workers, shared) | median 5.9, p90 13.8, max 87.1 min; 5.4 min CPU median | docs/validation/step4.md:175, 311-312 |
| Step 4 brute force (31 runs) | median 14.0, p90 96.3, max 104.0 min | docs/validation/step4.md:60, 174 |
| Step 5 plan, early stopping applied (31) | median 4.5, p90 10.9, mean 5.7, max 18.5 min | docs/validation/step5.md:85 |
| Step 5 plan, 2292 runs on 8 workers | 44.2 h (p90 46.2); with censoring 27.5 h (p90 28.1) | docs/validation/step5.md:95; docs/preregistration.md:224-225 |
| Step 6a (D_train 512, D_cal 256, D_test 256, D_shift 128) | batch wall 16.99 h; max per point 79.3 min (D_train 41.1, D_cal 57.2, D_test 79.3, D_shift 77.9 min); per-point median NOT FOUND in docs (per-point `wall_time_s` is in data/generated/D_*.csv) | docs/validation/step6b.md:14; docs/preregistration.md:333-334 |
| Step 6a status | D_train 432 css / 80 censored; D_cal 215/41; D_test 214/41/1 cap; D_shift 112/16; 0 failed | docs/validation/step6b.md:9-12 |
| Step 6c S_nom (900) | median 2.02, p90 4.42, max 120 min; batch 28.2 h (incl. hung attempt + 2 h retry); 837 css, 62 censored, 1 failed | docs/validation/step6c.md:13, 230 |
| Step 6c S_dist (240) | median 3.08, p90 5.57, max 9.27 min; batch 2.31 h | docs/validation/step6c.md:14, 232 |
| Step 7b (660 runs) | 4.9 h on 6 workers; blocks 1.549 / 1.569 / 1.517 h (210 each), cold spot checks 0.08069 / 0.08349 / 0.07235 h | docs/validation/step7b.md:14-19, 256 |
| Step 8a S_rep (405) | 3.738 h; 382 css, 23 censored, 0 failed | docs/validation/step8a.md:17-19 |

## 3. Layered-bed column used in Step 6

### 3.1 Geometry and wall (compact LSRE column, Lopes 2010 ch.5)

| item | value | source |
|---|---|---|
| bed length L | `bed_height: 0.267` [m] (Table 5.3 p.165) | data/params/compact_column_lopes2010.yaml:4, 7; src/h2psa/design.py:76 |
| internal diameter | `internal_diameter: 0.0500` [m] "DERIVED: sqrt(4 V / (pi L)) = 0.04999 m" | data/params/compact_column_lopes2010.yaml:10 |
| bed volume | `bed_volume: 5.24e-4` [m3] | data/params/compact_column_lopes2010.yaml:8 |
| wall thickness | `value: 0.003, range: [0.002, 0.005], units: m` (assumed) | data/params/compact_column_lopes2010.yaml:74 |
| wall density | `8000.0` kg m-3, stainless steel (assumed) | data/params/compact_column_lopes2010.yaml:75 |
| wall heat capacity | `500.0` J kg-1 K-1, stainless steel (assumed) | data/params/compact_column_lopes2010.yaml:76 |
| wall heat capacity per length Cw | rho_w Cp_w pi e (d + e) | src/h2psa/column.py:505-510, 63 |
| wall conductivity | `14.9` W m-1 K-1 (AISI 304) — enters only the U correlation, which Step 6 does not use | data/params/gas_transport_lopes2010.yaml:21; src/h2psa/transport.py:138; src/h2psa/design.py:101 |
| design-space statement | `column: {L: 0.267 m, d_wi: 0.05 m, wall: "stainless steel, 3 mm (assumed)", source: "Table 5.3"}` | configs/design_space.yaml:32 |
| ambient temperature | 303 K (heater, = feed T); `T_amb=c.T_feed` | data/params/compact_column_lopes2010.yaml:79; configs/design_space.yaml:35; src/h2psa/design.py:100 |
| thermocouples (not used in Step 6) | `[0.059, 0.151, 0.239]` m from inlet | data/params/compact_column_lopes2010.yaml:11 |

### 3.2 Adsorbent layers (HY2SEPS, Lopes 2010 ch.2-3)

Feed end = HY2SEPS-AC (fraction 1 - f_zeolite of L), product end = HY2SEPS-ZEO (src/h2psa/column.py:568-587; src/h2psa/design.py:100).

| item | HY2SEPS-AC | HY2SEPS-ZEO | source |
|---|---|---|---|
| bed porosity (used) | `bed_porosity_ch3: 0.394` (Table 3.10 p.94, 1.65 cm column; ASSUMED for compact column) | `0.391` (Table 3.13 p.101; ASSUMED) | hy2seps_ac_lopes2010.yaml:15; hy2seps_zeo_lopes2010.yaml:15; column.py:560; PROVENANCE.md:80 |
| particle (pellet) porosity | `0.566` | `0.503` | hy2seps_ac:10; hy2seps_zeo:11 |
| pellet diameter | `2.35e-3` [m] | `1.70e-3` [m] | hy2seps_ac:8; hy2seps_zeo:9 |
| particle radius | Rp = pellet_diameter / 2 (code rule; no radius value in YAML) | same | src/h2psa/column.py:556 |
| pellet length | `[3.0e-3, 4.0e-3]` | `[4.0e-3, 6.0e-3]` | hy2seps_ac:9; hy2seps_zeo:10 |
| particle density | `842.0` kg m-3 | `1126.0` kg m-3 | hy2seps_ac:11; hy2seps_zeo:12 |
| solid density | `1939.0` kg m-3 | `2267.0` kg m-3 | hy2seps_ac:12; hy2seps_zeo:13 |
| bulk density | NOT FOUND as a value (adsorbent mass computed as sum A dz (1 - eps) rho_p) | NOT FOUND | src/h2psa/design.py:103 |
| BET area | `4.808e5` m2 kg-1 | `4.339e5` m2 kg-1 | hy2seps_ac:13; hy2seps_zeo:14 |
| LDF factor macropore Omega_m | 8 (eq. 3.17 p.77) | 8 | hy2seps_ac:16; hy2seps_zeo:16 |
| LDF factor micropore/crystal Omega_c | 3 (p.98) | 15 (p.104) | hy2seps_ac:17; hy2seps_zeo:17 |
| tortuosity | 2.0 (p.96) | 2.0 (p.102) | hy2seps_ac:18; hy2seps_zeo:18 |
| particle specific area a_p | `value: 2.0` x 1/R_p (ASSUMED, "2/R_p = cylindrical extrudate") | same | compact_column_lopes2010.yaml:82-83; column.py:563 |
| solid heat capacity Cps | `880.0`, range `[700.0, 1100.0]` J kg-1 K-1, NOT reported, ASSUMED | `920.0`, range `[800.0, 1100.0]`, ASSUMED | hy2seps_ac:19-20; hy2seps_zeo:19-20; PROVENANCE.md:81 |
| zeolite type | — | "type undisclosed; similar to 4A" | hy2seps_zeo:2, 6 |

All rows: data/params/hy2seps_ac_lopes2010.yaml (hy2seps_ac) and data/params/hy2seps_zeo_lopes2010.yaml (hy2seps_zeo).

### 3.3 Isotherms (multisite Langmuir, Nitta; eq. 2.10 p.35)

Model: q/qmax = a K P (1 - q/qmax)^a, K = K_inf exp(-dH/(R T)), P in kPa (hy2seps_ac:22-27); mixture form `q_i/q_max,i = a_i K_i P_i (1 - sum_j q_j/q_max,j)^a_i` (src/h2psa/isotherms.py:303-304). Validity T 303-343 K, P 0-7 bar (hy2seps_ac:25; hy2seps_zeo:24). Units: q_max mol kg-1, K_inf kPa-1, minus_dH kJ mol-1. He does not adsorb (not in isotherm names; src/h2psa/column.py:196, 198). Heat of adsorption in the energy balance = isotherm minus_dH (src/h2psa/column.py:198).

| species | AC q_max | AC a | AC K_inf | AC -dH | ZEO q_max | ZEO a | ZEO K_inf | ZEO -dH |
|---|---|---|---|---|---|---|---|---|
| CO2 | 7.855 | 3.0 | 2.1278e-8 | 29.1 | 4.525 | 2.2 | 11.121e-8 | 36.0 |
| H2 | 23.57 | 1.0 | 7.6902e-8 | 12.8 | 9.954 | 1.0 | 52.024e-8 | 9.23 |
| CH4 | 6.733 | 3.5 | 7.9165e-8 | 22.7 | 4.976 | 2.0 | 17.793e-8 | 20.6 |
| CO | 9.063 | 2.6 | 2.6801e-8 | 22.6 | 3.828 | 2.6 | 1.5164e-8 | 29.8 |
| N2 | 5.891 | 4.0 | 23.428e-8 | 16.3 | 4.148 | 2.4 | 12.851e-8 | 20.4 |

Sources: AC Table 2.7 p.49, data/params/hy2seps_ac_lopes2010.yaml:26-33; ZEO Table 2.8 p.54, data/params/hy2seps_zeo_lopes2010.yaml:25-32.
Capacity factor phi multiplies q_max of every species in both layers (K, a kept): src/h2psa/column.py:547-559; configs/disturbances.yaml:6-8.

### 3.4 Kinetics (bi-LDF: macropore + micropore/crystal)

Measured Dc/rc2 are primary; Arrhenius refitted by least squares in log space (`arrhenius_fit`); printed (Dc0, Ea) kept for traceability only (src/h2psa/kinetics.py:1-19; src/h2psa/column.py:553-554; data/PROVENANCE.md:25-26, errata E4, E5). Units s-1 / kJ mol-1.

| species | AC Dc/rc2 303 / 323 / 343 K | AC printed Dc0/rc2, Ea | ZEO Dc/rc2 303 / 323 / 343 K | ZEO printed Dc0/rc2, Ea |
|---|---|---|---|---|
| CO2 | 3.35e-2 / 4.74e-2 / 6.97e-2 | 17.5, 15.8 | 3.41e-4 / 5.17e-4 / 7.84e-4 | 0.268, 18.0 |
| H2 | 2.31e-1 / 3.17e-1 / 3.73e-1 | 14.8, 10.4 | 4.97e-2 / 6.40e-2 / 8.53e-2 | 9.33, 11.6 |
| CH4 | 9.41e-3 / 1.64e-2 (corrected, E4) / 2.71e-2 | 81.5, 22.8 | 5.05e-3 / 1.13e-2 / 1.29e-2 | 36.9, 20.6 |
| CO | 5.79e-2 / 8.08e-2 / 1.31e-1 | 59.2, 17.5 | 2.34e-3 / 4.20e-3 / 8.85e-3 | 368.0, 28.7 |
| N2 | 6.44e-2 / 6.47e-2 / 8.98e-2 | 0.995, 7.03 | 1.13e-2 / 1.66e-2 / 2.24e-2 | 7.86, 14.9 |

Sources: AC Table 3.12 p.98, data/params/hy2seps_ac_lopes2010.yaml:35-41; ZEO Table 3.15 p.103, data/params/hy2seps_zeo_lopes2010.yaml:35-41. Refitted (D0, Ea) values: NOT FOUND as stored numbers (computed at run time).

Macropore: rate k = Omega_m Dp / Rp^2 (src/h2psa/column.py:199); Dp from eq. 5.17 (Bosanquet, tau = 2, no Knudsen) at the point's feed conditions; the quoted pp.97/103 values (AC H2 7.21e-5 ... ; ZEO H2 7.56e-5 ... m2 s-1 at 303 K) are a check only, not inputs (data/PROVENANCE.md:83; hy2seps_ac:43-48; hy2seps_zeo:43-48). Film: kf, Biot number eq. 5.10 (src/h2psa/column.py:200-201).

### 3.5 Transport and heat transfer

| item | value | source |
|---|---|---|
| h_w (gas-wall) | 55 W m-2 K-1, every layer (Table 5.9, group "1-4", p.177) | data/params/ac5ks_lopes2010.yaml:54, 57; src/h2psa/design.py:97, 101; data/PROVENANCE.md:87 |
| U (wall-ambient) | 50 W m-2 K-1, every layer (same source) | same as above |
| Dax | eq. 3.18: (0.45 + 0.55 eps) Dm + 0.35 Rp u_i, per layer, at the point's feed conditions; eq. 5.20 rejected | src/h2psa/transport.py:94-96, 127; src/h2psa/design.py:100; data/PROVENANCE.md:86 |
| lambda, kf, hf, Dp | correlations eqs 5.17-5.23 (Wakao-Funazkri), per layer, at the point's T_feed, P_high, feed flow, composition | src/h2psa/transport.py:106-141; src/h2psa/column.py:532-540, 577-585; src/h2psa/design.py:100 |
| scalar Dax / kf diffusivity | mole-fraction mean of species mixture diffusivities (ASSUMED) | data/PROVENANCE.md:85; src/h2psa/transport.py:121-122 |
| example values at experiment feed (5 bar, 303 K) | AC: Dax 1.41e-05, lambda 0.869, kf 0.0179, hf 156, Re 1.71; ZEO: Dax 1.26e-05, lambda 0.858, kf 0.0231, hf 202, Re 1.23; Dp H2/CO2/CH4/CO/N2 6.77e-06 / 5.75e-06 / 4.57e-06 / 4.5e-06 / 4.44e-06 (both) | docs/validation/step4.md:232-237 |
| gas properties | NIST Cp, viscosity at feed T (isobaric data at 5 bar) | data/params/gas_properties_nist.yaml:1-17; src/h2psa/column.py:209-214 |
| gas conductivity | modified Eucken + Bird eq. 9.3-17 (ASSUMED) | data/params/gas_transport_lopes2010.yaml:13-16; data/PROVENANCE.md:84 |

### 3.6 Feed

| item | value | source |
|---|---|---|
| nominal composition | `{H2: 0.79, CO2: 0.17, CH4: 0.021, CO: 0.012, N2: 0.007}` (Table 5.4 p.167) | data/params/compact_column_lopes2010.yaml:14-15 |
| temperature | `303.0` [K] | data/params/compact_column_lopes2010.yaml:16 |
| flow | `flow_rate_std: 5.00e-5` [m3 s-1] at 298 K and 1 bar "-> 2.018e-3 mol s-1" | data/params/compact_column_lopes2010.yaml:17, 54; src/h2psa/cycle.py:69-71 |
| pressure (nominal P_high) | `P_high: 5.0` bar | data/params/compact_column_lopes2010.yaml:21 |
| disturbed feed rule | "CO2 = (CH4 + CO + CO2 of the anchor) - y_CH4 - y_CO ..., N2 of the anchor, H2 = 1 - sum" | configs/disturbances.yaml:16; src/h2psa/design.py:39-52 |
| NETL dry anchor (not used in Step 6 blocks; all `anchor = "lopes"`) | 0.7502 H2 / 0.1896 CO2 / 0.0514 CH4 / 0.0050 CO / 0.0038 N2 | configs/disturbances.yaml:21-23; scripts/run_step6a.py:70 |
| initial bed state | He-filled at T_feed and P_high | src/h2psa/cycle.py:102-110; data/params/compact_column_lopes2010.yaml:25 |

### 3.7 Cycle emulation with one column

Step sequence (`Cycle.steps`): Feed, ED1..n, Blow, Purge, EPn..1, Press (src/h2psa/cycle.py:56-62). n = n_equalisations; kind "A" (1), "B" (3), "2" otherwise (src/h2psa/design.py:87-90).

| step | z = 0 (feed end) | z = L (product end) | pressure law | source |
|---|---|---|---|---|
| Feed | molar inflow (Danckwerts), feed y, T_feed | pressure P_high (product out) | constant | src/h2psa/cycle.py:132-133, 150-151 |
| ED_k | closed | pressure, outflow | exponential to P_inter[k], one beta_P | src/h2psa/cycle.py:145-147, 152-153 |
| Blow | pressure, outflow | closed | exponential to P_low | src/h2psa/cycle.py:154-155 |
| Purge | pressure P_low | pure H2 inflow at fixed molar flow | constant | src/h2psa/cycle.py:156-157 |
| EP_k | closed | pressure, inflow of the EP gas | exponential to P_inter[k] | src/h2psa/cycle.py:158-165 |
| Press | closed | H2 inflow | linear from step-start P to P_high (eq. 5.47) | src/h2psa/cycle.py:166-168 |

All inflows enter at the feed temperature (src/h2psa/cycle.py:18). Each exponential step starts from the pressure actually reached (src/h2psa/cycle.py:6-8).

| item | value | source |
|---|---|---|
| EP gas, Step 6 baseline | `ep_inflow: h2` (pure H2), "FROZEN BASELINE (step-3 closure, 2026-10-10)" | data/params/compact_column_lopes2010.yaml:31-35; src/h2psa/cycle.py:86; docs/preregistration.md:35 |
| EP gas, Step 7b scenario `ep_inflow_ed_gas` | `ep_gas = "ed"`: well-mixed gas collected in the matching ED step of the same cycle (ED1 -> EP1, ...), "a tank standing in for the partner bed" | src/h2psa/design.py:62-63; src/h2psa/cycle.py:11-14, 164, 226-228 |
| stored ED outlet history (time-resolved) | options `ed_direct` / `ed_reversed` (20 equal-mole bins); not used in Steps 6-8 | src/h2psa/cycle.py:14-16, 159-163, 171-180, 229 |
| beta_P | `0.102` s-1, one constant for ED, Blow, EP (fitted to author's sim 13, Fig. 5.20a) | data/params/compact_column_lopes2010.yaml:36, 96-98; configs/design_space.yaml:33 |
| equalisation pressures | P_k = P_high - k (P_high - P_low) / (n + 1), k = 1..n | configs/design_space.yaml:28; src/h2psa/design.py:89 |
| purge flow | purge_to_feed x (H2 fed per cycle) / t_purge, pure H2 at z = L | configs/design_space.yaml:29; src/h2psa/design.py:91-92 |
| fixed step times | `t_eq: 16 s, t_press: 21 s, t_purge: 40 s` (Table 5.4) | configs/design_space.yaml:36 |
| experiment cycle B (nominal reference) | steps `[Feed, ED1, ED2, ED3, Blow, Purge, EP3, EP2, EP1, Press]`; P 5.0 / 3.875 / 2.75 / 1.625 / 0.5 bar; times Feed 101, ED 16 each, Blow 201, Purge 40, EP 16 each, Press 21 s; cycle 459 s; purge/press gas hydrogen | data/params/compact_column_lopes2010.yaml:19-24 |
| experiment purge flow (fitted) | `0.000261` mol s-1, sd 3.9e-05 | data/params/compact_column_lopes2010.yaml:99-101 |
| adsorbent mass for productivity | mass of the layered bed (computed per point) | src/h2psa/design.py:103; scripts/run_batch.py:184-185 |
| performance | purity eq. 5.48 (feed components only), recovery eq. 5.49, productivity eq. 5.50 | src/h2psa/cycle.py:246-264; configs/design_space.yaml:38-44 |

### 3.8 Design space and disturbances

| variable | range | units | experiment value | source |
|---|---|---|---|---|
| t_feed | [40, 150] | s | 101 | configs/design_space.yaml:10; compact_column_lopes2010.yaml:22 |
| t_blow | [60, 200] | s | 201 | configs/design_space.yaml:12-13 |
| purge_to_feed | [0.02, 0.30] | mol H2 purge / mol H2 fed per cycle | 0.065 | configs/design_space.yaml:14-15 |
| P_high | [4.0, 8.5] | bar | 5.0 | configs/design_space.yaml:16 |
| P_low | [0.3, 1.5] | bar | 0.5 | configs/design_space.yaml:18-19 |
| f_zeolite | [0.0, 1.0], grid 30 (multiples of 1/30) | - | Step-4 baseline default 0.4 | configs/design_space.yaml:20-21; src/h2psa/design.py:93 |
| n_equalisations | {1, 2, 3} | - | 3 | configs/design_space.yaml:22-23 |
| feed_flow | [0.5, 1.5] | x 5.0e-5 m3/s at 298 K, 1 bar | 1.0 | configs/design_space.yaml:24-25 |
| phi (disturbance) | [0.70, 1.00], nominal 1.0 | - | — | configs/disturbances.yaml:6-8 |
| y_CH4 (disturbance) | [0.0168, 0.0231], nominal 0.021 | mol fraction, dry | — | configs/disturbances.yaml:9-11 |
| y_CO (disturbance) | [0.005, 0.015], nominal 0.012 | mol fraction, dry | — | configs/disturbances.yaml:12-14 |

Unit-cube mapping: continuous linear; f_zeolite = floor(31 u)/30 (capped at 30/30); n_eq = 1 + floor(3 u) (capped at 3) — docs/preregistration.md:39-41; scripts/run_step6a.py:54-62. Disturbances only vary in D_shift / S_dist; elsewhere nominal (scripts/run_step6a.py:63-69; docs/preregistration.md:37-38).

### Notes on sections 2-3: disagreements / inconsistencies found

1. Warm-start distance: Step 4 used Euclidean distance in (f_AC, t_feed/110 s, feed-flow ratio) (docs/validation/step4.md:87-89); Steps 5-8 use `design.scaled` over all 8 decision + 3 disturbance bounds (src/h2psa/design.py:106-121; scripts/run_batch.py:61-69). Evolution, not a conflict in Step 6.
2. `Cycle.ep_gas` dataclass default is `"ed"` (src/h2psa/cycle.py:49), but `experiment_cycle()` sets it from YAML `ep_inflow: h2` (src/h2psa/cycle.py:86; compact_column_lopes2010.yaml:31), so the Step 6 baseline is pure H2. Also the YAML comment says the pre-declared step-3 baseline was `ed` with beta_P = 0.1998 (compact_column_lopes2010.yaml:34-35, 88).
3. Bed porosity: compact column (AC5-KS) `bed_porosity: 0.404` (compact_column_lopes2010.yaml:9) vs. HY2SEPS layers in the same column 0.394 / 0.391 (assumed from ch.3 columns; hy2seps_*:15; PROVENANCE.md:80). The layered model uses 0.394 / 0.391.
4. Max per-point time S_nom: step6c table says max 120 min (docs/validation/step6c.md:13, includes the failed S_nom_0590 at the 2 h limit) vs. "The 899 finished S_nom points took at most 9.7 min" (docs/preregistration.md:334-335).
5. Workers: registered 8 (docs/preregistration.md:223; docs/validation/step5.md:78) vs. 6 actually used (docs/preregistration.md:230-235, 310-314; docs/validation/step6c.md:5; docs/validation/step7b.md:256).
6. `Cycle.kind` docstring says "A" (one equalisation) or "B" (three) (src/h2psa/cycle.py:35); with n = 2 design.build sets kind "2" (src/h2psa/design.py:88).

## 4. Validation numbers

Sources: `docs/validation/step2.md`, `step3.md`, `step4.md` (refs are `file:line`). The folders
`docs/validation/step2/`, `step3/`, `step4/` contain only PNG figures, **no CSVs** (CSV names cited in step3.md,
e.g. `fit_bcs_beta.csv` step3.md:453, `fit_bcs.csv` step3.md:688, are not in docs/ — NOT FOUND there).

Wording rule (CLAUDE.md): Lopes (2010) PSA performance numbers (81.6 % recovery, 99.98054 % purity, 195 ppm,
63 ppm CO, 101.1 mol/kg/day) are the author's **reference simulations** (sim 18), not measurements. Places where a
doc still calls them "experiment"/"measured" are flagged **[WORDING]**.

---

### 4.1 Step 2 — breakthrough (column model), runs 1-5

Setup: N = 60 finite volumes, BDF (step2.md:5); experiment = Lopes (2010) Figs 5.15-5.17, digitised (step2.md:6).
Metrics computed: t10 and t50 breakthrough times (relative error vs measured), thermocouple peak rise / peak time /
RMS. No RMSE of the breakthrough curves themselves is reported (NOT FOUND). H2 not in the table: first H2 sample
taken after H2 broke through (step2.md:227).

**Breakthrough times, final model (gate |error| <= 10 %)** — step2.md:229-248

| run | species | level | sim_s | exp_s | rel_err | within 10 % | ref |
|---|---|---|---|---|---|---|---|
| 1 | CO2 | t10 | 636 | 617 | +3.2% | True | step2.md:231 |
| 1 | CO2 | t50 | 707 | 645 | +9.6% | True | step2.md:232 |
| 2 | CO2 | t10 | 898 | 928 | -3.2% | True | step2.md:233 |
| 2 | CO2 | t50 | 1007 | 997 | +1.0% | True | step2.md:234 |
| 3 | CO2 | t10 | 413 | 395 | +4.6% | True | step2.md:235 |
| 3 | CO2 | t50 | 429 | 417 | +2.8% | True | step2.md:236 |
| 4 | CO2 | t10 | 669 | 641 | +4.4% | True | step2.md:237 |
| 4 | CO2 | t50 | 745 | 704 | +5.8% | True | step2.md:238 |
| 4 | CO | t10 | 245 | 243 | +1.0% | True | step2.md:239 |
| 4 | CO | t50 | 285 | 280 | +1.8% | True | step2.md:240 |
| 5 | CO2 | t10 | 688 | 677 | +1.6% | True | step2.md:241 |
| 5 | CO2 | t50 | 775 | 723 | +7.2% | True | step2.md:242 |
| 5 | CH4 | t10 | 506 | 421 | +20.3% | **False** | step2.md:243 |
| 5 | CH4 | t50 | 539 | 488 | +10.4% | **False** | step2.md:244 |
| 5 | CO | t10 | 246 | 226 | +8.8% | True | step2.md:245 |
| 5 | CO | t50 | 294 | 275 | +7.1% | True | step2.md:246 |
| 5 | N2 | t10 | 185 | 163 | +13.2% | **False** | step2.md:247 |
| 5 | N2 | t50 | 219 | 223 | -1.9% | True | step2.md:248 |

Roles: CO2 times of runs 1-3 = "fit" (calibration objective), runs 4-5 = "held out" (step2.md:39-60).
Before the decisions: 16/18 within 10 % (step2.md:29); run 5 CH4 t10 -53.3 %, t50 -13.8 % with Table 5.7
kinetics (step2.md:55-56).

Sensitivity cases, count within 10 %: kinetics (b) Table 4.10 all species 14/18; mixing iast 15/18;
noncompetitive_h2 14/18; a_p = 3/R_p 15/18; a_p = 2/R_p + 2/L_p 15/18; thermal bounded fit (not adopted) 16/18
(step2.md:250). Per-item values step2.md:252-271.

**Thermocouples, final model** (qualitative gate: rise above 323 K within 30 %, peak time within 20 %, B -> M -> T
ordering; sigma_T = 0.85 K) — step2.md:275-295

| run | tc | sim_peak_K | exp_peak_K | rise_rel_err | sim_time_s | exp_time_s | time_rel_err | rms_K | ok | ref |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Bottom | 346.3 | 354.9 | -27% | 149 | 165 | -10% | 2.74 | True | step2.md:281 |
| 1 | Middle | 343.2 | 349.9 | -25% | 379 | 375 | +1% | 2.04 | True | step2.md:282 |
| 1 | Top | 341.6 | 346.5 | -21% | 622 | 588 | +6% | 2.47 | True | step2.md:283 |
| 2 | Bottom | 333.0 | 340.0 | -41% | 230 | 257 | -11% | 2.06 | False | step2.md:284 |
| 2 | Middle | 331.5 | 336.2 | -36% | 568 | 570 | -0% | 1.82 | False | step2.md:285 |
| 2 | Top | 330.8 | 335.0 | -35% | 910 | 901 | +1% | 2.22 | False | step2.md:286 |
| 3 | Bottom | 366.1 | 373.1 | -14% | 90 | 124 | -27% | 6.19 | False | step2.md:287 |
| 3 | Middle | 364.2 | 367.4 | -7% | 234 | 251 | -7% | 5.21 | True | step2.md:288 |
| 3 | Top | 363.2 | 362.6 | +2% | 390 | 396 | -2% | 3.97 | True | step2.md:289 |
| 4 | Bottom | 343.9 | 354.3 | -33% | 160 | 176 | -9% | 3.18 | False | step2.md:290 |
| 4 | Middle | 340.7 | 348.3 | -30% | 405 | 390 | +4% | 2.78 | False | step2.md:291 |
| 4 | Top | 339.1 | 344.9 | -27% | 661 | 618 | +7% | 2.67 | True | step2.md:292 |
| 5 | Bottom | 342.1 | 350.7 | -31% | 167 | 185 | -10% | 2.50 | False | step2.md:293 |
| 5 | Middle | 338.7 | 343.9 | -25% | 426 | 405 | +5% | 1.58 | True | step2.md:294 |
| 5 | Top | 337.0 | 341.0 | -22% | 694 | 654 | +6% | 2.22 | True | step2.md:295 |

Summary: ordering correct in all runs: True; 8 of 15 peaks within ±30% rise and ±20% time; rise error median -27%,
range -41% to +2% (step2.md:32-33). RMS = RMS of T_sim - T_exp over 0-2500 s (step2.md:62).

**Gates (step 2)**

| gate | value | limit | result | ref |
|---|---|---|---|---|
| grid t50 N=60->120 | 3.03e-03 | 1e-02 | True (PASS) | step2.md:218 |
| linear isotherm vs t_st (t_st = 646.1 s) | 5.93e-03 | 1e-02 | True (PASS) | step2.md:219, :223 |
| mole balance | 6.01e-08 | 1e-03 | True (PASS) | step2.md:220 |
| breakthrough times within 10 % | 15/18 | all (\|err\| <= 10%) | **FAIL** (run 5 CH4 t10 +20%, CH4 t50 +10%, N2 t10 +13%) | step2.md:29-30 |
| temperature peaks (qualitative) | 8 of 15 peaks; ordering True | rise 30 %, time 20 %, ordering | not labelled pass/fail in summary (per-peak `qualitative_ok` in table above) | step2.md:32-33, :281-295 |

Decision: run-5 CH4 t10 (+20 %) and N2 t10 (+13 %) accepted as known limitations; N2 is the purity-limiting
impurity, released 10-13 % late in run 5, so purity predictions expected **optimistic** (step2.md:364-366).
Thermal calibration not adopted: Cps = 1061 J/kg/K, Cw = 317 J/m/K on lower bound, correlation -0.94,
chi2/obs 13.2 (step2.md:31, :104-109, :368-370).

Other step-2 numbers: model vs author's gPROMS curves, e.g. run 5 N2 t10 184.9 vs 174.7 s (+5.8%), CH4 t10 506.0 vs
478.5 s (+5.8%) (step2.md:302-336). Author's own model misses CH4 t10 by +14% (t10 478 s vs 421 s measured)
(step2.md:128-129). Runtime 9 s per breakthrough (6000 s) at N = 60 with Taqvi-LeVan, 218 s with IAST
(step2.md:34-35).

[WORDING] step2.md:135: "PSA experiment (held out, not simulated here): 99.9805 % H2, i.e. 195 ppm impurities, of
which the thesis attributes 63 ppm to CO (p.191)" — these are the author's reference-simulation values (sim 18),
not measured.

---

### 4.2 Step 3 — PSA cycle (corrected model = frozen baseline)

Corrected model: E7 kinetics, `taqvi_levan`, assumed thermal values, beta_P = 0.102 s-1, EP inflow = pure H2
(step3.md:21-22, :46-49).

#### 4.2.1 Verdict / gates (corrected model) — step3.md:53-63, :107-115

| check | value | limit | result | ref |
|---|---|---|---|---|
| CSS reached, all corrected-model cases | 21/21 | all | PASS | step3.md:76 |
| per-cycle mole balance, worst | 1.2e-06 | < 1e-3 | PASS | step3.md:77 |
| grid N 30 -> 60: recovery | 0.031 pp | < 0.3 pp | PASS | step3.md:78 |
| grid N 30 -> 60: impurity ppm | 8.4 % | < 10 % | PASS | step3.md:79 |
| Spearman, recovery (18 sims) | 1.000 | >= 0.9 | PASS | step3.md:109 |
| Spearman, ln(impurity ppm) (18 sims) | 0.986 | >= 0.8 | PASS | step3.md:110 |
| Spearman, recovery (17 sims, Table 5.10) | 1.000 | >= 0.9 | PASS | step3.md:111 |
| Spearman, ln(impurity ppm) (17 sims, Table 5.10) | 0.983 | >= 0.8 | PASS | step3.md:112 |
| pairwise effects, same sign (Table 5.10 pairs) | 36/38 | all | **FAIL** | step3.md:113 |
| impurity within 2x of author | 13/17 (sim 5 not rated: author 0 ppm) | all | **FAIL** | step3.md:56, :114 |
| recovery within 2 pp of author | 0/18; offset -7.4 to -2.9 pp (mean -4.21, sd 1.04) | all | **FAIL** (accepted as documented discrepancy) | step3.md:57, :115 |
| 3a. pressure, cycle 1 (Fig. 5.18a) | RMS 0.25 bar overall | no limit declared | reported | step3.md:59 |
| 3b. H2 outflow per step, cycles 1 and 37 | 3/6 evaluable steps | +-10 % | **FAIL** | step3.md:60 |
| 3b. CO2 front in feed step | no front in experiment or model | +-10 % | not evaluable | step3.md:61 |
| 3c. H2 recovery at cycle 37 from measured streams | streams do not close H2 balance | +-3 pp | not evaluable | step3.md:62 |
| 3d. thermocouples, cycles 1-10 | 15/30 peaks, ordering 2/10 cycles | rise 30 %, peak time 20 %, ordering | **FAIL** | step3.md:63 |

Closure: step 3 closed with documented limitations, none coding bugs (step3.md:7); 3b FAIL, 3d FAIL, 3c data
limitation (step3.md:13-15); blowdown sign misses 8 -> 9 and 11 -> 12 stay recorded as failures (step3.md:11-12).

#### 4.2.2 Measured cyclic data (validation)

**3a. Pressure, cycle 1, RMS per step type [bar]** — step3.md:219-226

| type | n | rms_bar |
|---|---|---|
| Feed | 97.0 | 0.004 |
| ED | 46.0 | 0.731 |
| Blow | 197.0 | 0.094 |
| Purge | 40.0 | 0.201 |
| EP | 48.0 | 0.141 |
| Press | 20.0 | 0.051 |

Overall RMS 0.25 bar (step3.md:59); band table gives p_rms_bar 0.254 (step3.md:342). Pressure law limitation:
RMS 0.7 bar in ED (step3.md:363-364).

**3b. H2 leaving the bed per step (evaluable steps only; ED/EP/Press: no samples)** — step3.md:233-254

| cycle | step | n | window | measured_mmol | model_mmol | rel_err | result | ref |
|---|---|---|---|---|---|---|---|---|
| 1 | Feed | 6 | 19-97 s | 64.50 | 37.96 | -41.1% | FAIL | step3.md:235 |
| 1 | Blow | 3 | 173-219 s | 3.95 | 3.93 | -0.5% | PASS | step3.md:239 |
| 1 | Purge | 3 | 359-379 s | 4.73 | 5.02 | +6.3% | PASS | step3.md:240 |
| 37 | Feed | 4 | 40-97 s | 97.79 | 96.10 | -1.7% | PASS | step3.md:245 |
| 37 | Blow | 7 | 169-279 s | 6.54 | 4.83 | -26.2% | FAIL | step3.md:249 |
| 37 | Purge | 3 | 359-379 s | 3.81 | 4.86 | +27.6% | FAIL | step3.md:250 |

CO2 at z = L in feed step: threshold 0.034 mmol/s; exp max 0.0043, model max 6.53e-21 (cycle 1) / 2.37e-06
(cycle 37); no front in either (step3.md:258-261).

**3c. Partial H2 balance, cycle 37** — step3.md:281-294: H2 fed 161.02 mmol (set point); product 97.79 measured vs
96.10 model (window); model recovery at cycle 37 (eq. 5.49): 78.40 % (author's simulated value 81.6 %)
(step3.md:294). Experimental recovery cannot be formed (step3.md:279).

**3d. Thermocouples, cycles 1-10, corrected model** (rise above T_feed = 303 K) — step3.md:303-334

| cycle | Bottom rise err | Middle rise err | Top rise err | ok (B/M/T) | ref |
|---|---|---|---|---|---|
| 1 | -15% | -27% | -57% | T/T/F | step3.md:305-307 |
| 2 | +13% | +26% | -68% | T/T/F | step3.md:308-310 |
| 3 | +16% | +57% | -64% | T/F/F | step3.md:311-313 |
| 4 | +16% | +10% | -61% | T/T/F | step3.md:314-316 |
| 5 | +12% | -10% | -62% | T/T/F | step3.md:317-319 |
| 6 | +7% | -24% | -59% | T/T/F | step3.md:320-322 |
| 7 | +1% | -38% | -56% | T/F/F | step3.md:323-325 |
| 8 | +7% | -42% | -54% | T/F/F | step3.md:326-328 |
| 9 | +14% | -45% | -52% | T/F/F | step3.md:329-331 |
| 10 | +5% | -51% | -54% | T/F/F | step3.md:332-334 |

Top thermocouple rise 52-68 % low
in every cycle (Middle -51 to +57 %); Bottom passes in all 10 cycles (step3.md:367-368).

**Cycle 37 thermocouples, outlet flow RMS per species** — corrected model: NOT FOUND (only reported for the
first-comparison, pre-declared baseline model, below).

First-comparison baseline (pre-correction model; step3.md:724-781, "shape checks, stated errors, no pass/fail"):

| window | tc | n | rms_K | bias_K | swing_exp_K | swing_model_K | ref |
|---|---|---|---|---|---|---|---|
| cycles 1-10 | Bottom | 1035 | 1.16 | -0.83 | 8.5 | 9.8 | step3.md:756 |
| cycles 1-10 | Middle | 1148 | 0.74 | +0.27 | 4.8 | 2.3 | step3.md:757 |
| cycles 1-10 | Top | 1148 | 1.07 | +0.06 | 6.5 | 2.4 | step3.md:758 |
| cycle 1 | Bottom | 115 | 0.89 | -0.79 | 3.5 | 3.1 | step3.md:759 |
| cycle 1 | Middle | 115 | 0.79 | +0.58 | 3.3 | 2.2 | step3.md:760 |
| cycle 1 | Top | 115 | 1.11 | -0.75 | 4.7 | 2.2 | step3.md:761 |
| cycle 37 | Bottom | 115 | 1.40 | -1.04 | 7.8 | 8.9 | step3.md:762 |
| cycle 37 | Middle | 115 | 0.52 | -0.30 | 2.9 | 2.0 | step3.md:763 |
| cycle 37 | Top | 115 | 0.73 | -0.05 | 4.1 | 2.2 | step3.md:764 |

Outlet flows (first-comparison baseline, Fig. 5.19 main panel, rms_err / max_abs_err): cycle 1 H2 0.366 / 1.244,
CO2 0.043 / 0.125, CH4 0.011 / 0.029, CO 0.007 / 0.019, N2 0.005 / 0.012; cycle 37 H2 0.123 / 0.403, CO2 0.065 /
0.165, CH4 0.014 / 0.033, CO 0.008 / 0.016, N2 0.007 / 0.012 (step3.md:731-744; zoom panel step3.md:736-739,
:745-748).
Units of these flow errors: not stated in the table (NOT FOUND).
Pressure cycle 1 per step, first-comparison baseline: ED1 0.534, ED2 0.454, ED3 0.464, Blow 0.108, EP1 0.499 bar,
etc. (step3.md:770-781).

#### 4.2.3 Code-to-code verification vs Lopes (2010) reference simulations (corrected model)

Sims 1-17 = Table 5.10; sim 18 = experiment cycle, author's simulated values p.256 (step3.md:83). — step3.md:86-105

| sim | cycle | P_high | T | rec_ref | rec_model | d_rec_pp | ppm_ref | ppm_model | ppm_ratio | CO_ref | CO_model | imp. | rec. | ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 5 | 303 | 70.96 | 66.95 | -4.01 | 79.30 | 95.54 | 1.20 | 15.2 | 41.6 | PASS | FAIL | step3.md:88 |
| 2 | A | 5 | 303 | 68.68 | 64.07 | -4.61 | 5.90 | 14.08 | 2.39 | 0.705 | 6.09 | FAIL | FAIL | step3.md:89 |
| 3 | B | 5 | 303 | 82.40 | 79.54 | -2.86 | 444.50 | 351.91 | 0.79 | 160 | 155 | PASS | FAIL | step3.md:90 |
| 4 | B | 5 | 303 | 79.98 | 76.67 | -3.31 | 73.50 | 66.20 | 0.90 | 18 | 28.3 | PASS | FAIL | step3.md:91 |
| 5 | A | 10 | 303 | 47.10 | 39.73 | -7.37 | 0.00 | 0.11 | – | 0.000199 | 0.0754 | – (author 0 ppm) | FAIL | step3.md:92 |
| 6 | A | 10 | 303 | 57.17 | 51.12 | -6.05 | 0.30 | 3.00 | 10.00 | 0.0477 | 1.78 | FAIL | FAIL | step3.md:93 |
| 7 | B | 10 | 303 | 67.50 | 62.83 | -4.67 | 0.20 | 1.69 | 8.43 | 0.0262 | 1.01 | FAIL | FAIL | step3.md:94 |
| 8 | B | 10 | 303 | 74.17 | 70.33 | -3.84 | 20.50 | 33.43 | 1.63 | 5.01 | 16.7 | PASS | FAIL | step3.md:95 |
| 9 | B | 10 | 303 | 74.34 | 70.33 | -4.01 | 25.60 | 31.82 | 1.24 | 6.41 | 16 | PASS | FAIL | step3.md:96 |
| 10 | B | 10 | 303 | 75.77 | 72.06 | -3.71 | 88.90 | 92.72 | 1.04 | 26 | 43.1 | PASS | FAIL | step3.md:97 |
| 11 | B | 10 | 303 | 72.55 | 68.65 | -3.90 | 12.60 | 24.92 | 1.98 | 2.95 | 12.9 | PASS | FAIL | step3.md:98 |
| 12 | B | 10 | 303 | 72.77 | 68.69 | -4.08 | 18.60 | 24.21 | 1.30 | 4.45 | 12.7 | PASS | FAIL | step3.md:99 |
| 13 | B | 10 | 303 | 72.88 | 68.76 | -4.12 | 20.10 | 24.61 | 1.22 | 4.85 | 12.9 | PASS | FAIL | step3.md:100 |
| 14 | B | 10 | 303 | 76.14 | 72.19 | -3.95 | 129.90 | 94.66 | 0.73 | 37.4 | 44.8 | PASS | FAIL | step3.md:101 |
| 15 | B | 10 | 323 | 70.26 | 65.93 | -4.33 | 1.40 | 5.06 | 3.61 | 0.34 | 3.25 | FAIL | FAIL | step3.md:102 |
| 16 | B | 10 | 323 | 78.48 | 74.79 | -3.69 | 280.80 | 202.13 | 0.72 | 112 | 106 | PASS | FAIL | step3.md:103 |
| 17 | B | 10 | 323 | 76.34 | 72.31 | -4.03 | 69.10 | 52.05 | 0.75 | 27.1 | 29 | PASS | FAIL | step3.md:104 |
| 18 | B | 5 | 303 | 81.60 | 78.43 | -3.17 | 195.00 | 143.90 | 0.74 | 63 | 63.5 | PASS | FAIL | step3.md:105 |

(`_ref` = author's simulated values; columns "rec_author"/"ppm_author"/"co_author" in the doc.) Impurity ppm of
the author = 1e6 (1 - purity) (step3.md:831). Purity of the reference sims in % is not tabulated in step3.md beyond
sim 18 (99.98054 %, step3.md:32; 99.9805, step3.md:716).

- Recovery offset not constant: cycle A 5 bar -4.6 to -4.0 pp; cycle A 10 bar -7.4 to -6.0 pp; cycle B 5 bar
  -3.3 to -2.9 pp; cycle B 10 bar -4.7 to -3.7 pp (all 18: mean -4.21, sd 1.04 pp) (step3.md:162).
- All 19 recovery effects have the author's sign; size overstated by factor 1.13 (median; range 1.05-1.26 over the
  16 effects above 1 pp) (step3.md:163-165).
- Impurity failures sims 2, 6, 7, 15 (author 5.9, 0.3, 0.2, 1.4 ppm; ours 14.1, 3.0, 1.7, 5.1 ppm), all below
  6 ppm on the author's side (step3.md:169-170).
- Sim 18: 144 ppm and 63.5 ppm CO vs author's simulated 195 and 63 ppm (step3.md:171-172).
- Sign failures: 8 -> 9 (blow 200->150 s, impurity: author +5.1, ours -1.6); 11 -> 12 (blow 150->100 s: author
  +6.0, ours -0.7) (step3.md:144, :150, :172).
- Gas holdup: model releases 27-50 % more H2 per depressurisation step than the author's model (step3.md:359-361);
  sim 13 per step ratio ED1 1.31, ED2 1.42, ED3 1.50, Blow 1.27 (step3.md:190-193).

First comparison (pre-declared baseline, superseded; step3.md:371-380, :800-829): Spearman recovery 0.980 PASS,
ln(impurity) 0.841 PASS, same sign 37/38 FAIL (step3.md:807-809); impurity 7-1000x the author's in every
simulation (step3.md:629). Gate 2 of that comparison FAIL: CSS 81.86 % recovery, 1825 ppm, 674.9 ppm CO
(step3.md:714); limits recovery +-3 pp of 81.6 %, impurity within factor 2 of 195 ppm, CO within factor 3 of 63 ppm
(step3.md:718-719).
[WORDING] step3.md:716 row labelled "experiment" (81.60 / 195 / 63.0 / 99.9805 / 101.1); step3.md:620-622
"experiment 81.6 %", "measured 195 ppm", "measured 63 ppm"; step3.md:613 "First experiment comparison" — these are
reference-simulation values (the doc itself corrects this at step3.md:37-40, :373-374).

#### 4.2.4 Uncertainty band (declared assumptions) — corrected model, step3.md:340-351

| case | p_rms_bar | h2_steps_pass | T_peaks_ok | rise_err_median | ppm_css | rec_css | rec_c37 | ref |
|---|---|---|---|---|---|---|---|---|
| cor_exp (baseline, EP = pure H2) | 0.254 | 3/6 | 15/30 | -25% | 143.9 | 78.43 | 78.40 | step3.md:342 |
| cor_exp_ep_ed (EP inflow = ED gas) | 0.254 | 3/6 | 15/30 | -25% | 813.5 | 78.41 | 78.38 | step3.md:343 |
| cor_exp_kinetics_table4.10 | 0.254 | 3/6 | 15/30 | -24% | 46.4 | 78.40 | 78.37 | step3.md:344 |
| cor_exp_noncompetitive_h2 | 0.254 | 3/6 | 15/30 | -26% | 125.9 | 75.73 | 75.72 | step3.md:345 |
| cor_exp_thermal_high | 0.254 | 3/6 | 12/30 | -39% | 126.6 | 78.19 | 78.16 | step3.md:346 |
| cor_exp_thermal_low | 0.254 | 3/6 | 6/30 | -10% | 169.2 | 78.73 | 78.69 | step3.md:347 |

h2_max_abs_err 41% / 41% / 41% / 45% / 42% / 41% and product_window_err -1.7% / -1.8% / -1.7% / -3.0% / -1.8% / -1.6% (rows in table order); T_order_ok 2/10 in every row (step3.md:342-347).
EP inflow: pure H2 vs ED gas moves experiment-cycle impurity from 144 to 813 ppm at the same recovery
(step3.md:16-17, :357-358). Gas holdup: declared assumption; adsorbed-volume variant changes H2 released per step
by less than 1.5 %, not carried as a band member (step3.md:206-213).
First-comparison band (pre-correction): baseline 1825 ppm; EP gas H2 352; Table 4.10 1266; noncompetitive_h2 2076
(rec 79.19); thermal low/high 2017 / 1687; beta_P low/high 1064 / 3024 ppm (step3.md:787-794).

---

### 4.3 Step 4 — layered bed, trends (gate 4)

README.md:23 names the step-4 gate "Trends of Yang (2008)"; within step4.md only trend e is explicitly labelled
"Yang 2008" (step4.md:255); the literature source of trends a-d is not stated in step4.md (NOT FOUND).
Trends declared before running (step4.md:241-247). Overall: a, c, d reproduced; b, e NOT reproduced, no tuning
(step4.md:40, :22-25).

| trend | result | numbers | ref |
|---|---|---|---|
| a. layered purer than AC only at equal recovery | reproduced | f_AC 0.8: 81.2 vs AC only 113 ppm at 71.5 %; 0.6: 55.6 vs 119 ppm at 71.7 %; 0.4: 35.3 vs 125 ppm at 71.9 %; 0.2: 39.1 vs 133 ppm at 72.1 % | step4.md:251 |
| b. limiting impurity of AC only is N2 or CO; zeolite reduces it | **not reproduced** | AC only (t_feed 101 s): CO2 1.31, CH4 323, CO 89.7, N2 137 ppm -> limiting CH4; f_AC 0.6: CH4 175 ppm | step4.md:252 |
| c. interior optimum zeolite fraction (recovery at 99.99 %) | reproduced | f_AC 1.0: 71.0 %; 0.8: 72.2 %; 0.6: 73.6 %; 0.4: 74.8 %; 0.2: 74.3 %; 0.0: not reached | step4.md:253 |
| d. longer feed / higher feed velocity: lower purity, higher recovery | reproduced | feed time monotone for all fractions; feed flow x0.8/1.0/1.2 (f_AC 0.6): ppm 40.4/434/2.07e+03, recovery 71.6/78.2/82.8 | step4.md:254 |
| e. Yang 2008: >= 99.99 % with 60-80 % recovery at 3.1-5.2 cm/s | **not reproduced** | 3.14 cm/s: 0.31 ppm, 29.5 %; 3.90: 5.19 ppm, 43.6 %; 4.70: 44.1 ppm, 53.6 %; 5.20: 123 ppm, 58.2 % | step4.md:255 |

Explanations: b — CH4 limits the AC-only bed (323 ppm of 550 at t_feed 101 s), consistent with CH4 having the
slowest micropore kinetics on HY2SEPS-AC (step4.md:22-23, :48-50). e — product pressurisation takes 42-70 % of the
feed H2 in the single-column emulation of Yang's 4-bed 10-step cycle (step4.md:23-24, :50-51); press_in 0.695 /
0.558 / 0.461 / 0.416 mol per mol H2 fed at 3.14 / 3.90 / 4.70 / 5.20 cm/s (step4.md:262-265). Limitation for the
paper: single-column emulation gives absolute recoveries below industrial multi-bed values (step4.md:24-25, :51-52).

Other step-4 gates (step4.md:35-41, :102-109, :188-189, :197-210):

| gate | value | limit | result | ref |
|---|---|---|---|---|
| 1. CSS reached | 33/33 | all | PASS | step4.md:104 |
| 1. mole balance, worst cycle | 6.6e-07 | < 1e-3 | PASS | step4.md:105 |
| 1. grid N 30 -> 60, f_AC 0.6: recovery / impurity | 0.007 pp / 7.9 % | < 0.3 pp / < 10 % | PASS | step4.md:106-107 |
| 1. accelerated vs brute-force CSS | 14/14, ppm 25-1766 | ppm 1e-3 rel., recovery 0.05 pp | PASS | step4.md:109 |
| 2. AC5-KS limit = frozen step-3 baseline | 78.429 % / 143.92 vs 143.90 ppm (+0.02%) | – | PASS | step4.md:188 |
| 3. correlations vs Tables 5.7/5.9 (factor 1.5) | Dax (5.20) 51.23-54.56; hw (5.31) 1.78-7.75; U (5.29) 0.06-0.07; Dax (3.18) 1.15-1.24 | factor 1.5 | FAIL as pre-declared (5.20, 5.31, 5.29); resolved: Dax eq. 3.18 PASS (adopted), h_w/U from apparatus | step4.md:39, :199-205 |
| runtime per CSS | median 5.9, p90 13.8, max 87.1 min | <= 10 min | target holds for median only (p90 accepted) | step4.md:41, :30-31 |

# Sections 5 and 7: exact values and the thesis page check

Repo: /Users/izbassar/Documents/Projects/h2-psa-digital-twin (all paths below are relative to it). Read-only extraction, 2026-10-15.

## 5. Exact values

### 5.1 Two-sided 95 % Clopper-Pearson CIs of H_B1 and H_B2

From `docs/validation/step8a/pooled.csv` (header on line 1: `group,n,k,rate,ci_lower,ci_upper,failed,sensitivity_n,...`):

| hypothesis | group | file:line | n | k | rate | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|
| H_B1 | GP_L | docs/validation/step8a/pooled.csv:2 | 135 | 1 | 0.007407407407407408 | 0.0001875217340034464 | 0.040579359116086886 |
| H_B2 | GBM_L | docs/validation/step8a/pooled.csv:3 | 135 | 130 | 0.9629629629629629 | 0.915685422235369 | 0.9878666682525633 |

The sensitivity columns on the same lines are identical, because failed = 0.

As shown in `docs/validation/step8a.md` (caption on line 33: "Pooled violation rates (Clopper-Pearson 95 %):"):
- docs/validation/step8a.md:37 `| GP_L | 1/135 = 0.007 [0.000, 0.041] | 0 | 1 | 135 |`
- docs/validation/step8a.md:38 `| GBM_L | 130/135 = 0.963 [0.916, 0.988] | 0 | 130 | 135 |`

Confirmation that these are two-sided 95 % Clopper-Pearson intervals:
- `src/h2psa/step6_metrics.py:45-51` `clopper_pearson(k, n, alpha=0.05)` returns
  `(0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1), 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k))`.
  So the interval is equal-tailed with alpha/2 = 0.025 in each tail, which makes it two-sided 95 %.
- `src/h2psa/step6_metrics.py:76-78` `_rate(k, n)` calls `clopper_pearson(k, n)` with the default alpha and fills `ci_lower`/`ci_upper`.
- `src/h2psa/step6_metrics.py:89-94` `violation_summary(frame)` returns `**_rate(k, len(observed))` (k = count of z > 2.0, censored z set to 3.0, failed points excluded) plus `sensitivity_*` from `_rate(k + failed, len(frame))`.
- `scripts/analyse_step8a.py:85` `pooled = pd.DataFrame([{"group": g, **violation_summary(frame.loc[frame["group"] == g])} for g in GROUPS])`. This is the table written to pooled.csv.
- `scripts/analyse_step8a.py:156-157` formats the report string `f"{k}/{n} = {r:.3f} [{a:.3f}, {b:.3f}]"` from `ci_lower`/`ci_upper`.
- Pre-registered definition: docs/preregistration.md:127 "Violation rate of an arm: fraction of its re-simulated points with z > 2; two-sided Clopper-Pearson 95 % CI."
- Do not confuse these with the one-sided bound. `cp_upper` (`src/h2psa/step6_metrics.py:54-55`, `beta.ppf(1 - alpha, ...)`) is the one-sided upper bound used in the H_B1 test at the Holm level: docs/validation/step8a/hypotheses.csv:2 `cp_upper_at_level` = 0.03465622471536215 (shown as 0.03466 in docs/validation/step8a.md:30). That value is not the 95 % two-sided CI.

### 5.2 D_test coverage of the global CP variant (G), all 12 G arms

Source: `docs/validation/step6b/gate.csv` (header line 1: `arm,surrogate,n_train,variant,alpha,n,covered,coverage,cp_lower,cp_upper,cp_one_sided_upper,gate_pass,failed`). The report table is in `docs/validation/step6b.md` under "## Per-arm test gates and fixed bands" (line 64), header on line 66. n = 256 D_test points in every arm, failed = 0, gate_pass = True.

| arm | surrogate | n_train | alpha | covered/256 | coverage (gate.csv) | cp_lower | cp_upper | cp_one_sided_upper | gate.csv line | step6b.md line (rounded) |
|---|---|---|---|---|---|---|---|---|---|---|
| GP_n128_G_a0.10 | GP | 128 | 0.1 | 236 | 0.921875 | 0.8819198752154976 | 0.9516277084934217 | 0.9476259476863097 | 2 | 68: 0.9219 [0.8819, 0.9516] |
| GP_n256_G_a0.10 | GP | 256 | 0.1 | 237 | 0.92578125 | 0.8865230711884693 | 0.9547262489463787 | 0.9508562537719669 | 4 | 70: 0.9258 [0.8865, 0.9547] |
| GP_n512_G_a0.10 | GP | 512 | 0.1 | 228 | 0.890625 | 0.8458124702806946 | 0.9260797447946754 | 0.9211417037295324 | 6 | 72: 0.8906 [0.8458, 0.9261] |
| GP_n128_G_a0.05 | GP | 128 | 0.05 | 251 | 0.98046875 | 0.9550116788836257 | 0.9936285795451942 | 0.9922733511391899 | 8 | 74: 0.9805 [0.9550, 0.9936] |
| GP_n256_G_a0.05 | GP | 256 | 0.05 | 246 | 0.9609375 | 0.9293377140998706 | 0.9811117242805399 | 0.9786567696222699 | 10 | 76: 0.9609 [0.9293, 0.9811] |
| GP_n512_G_a0.05 | GP | 512 | 0.05 | 244 | 0.953125 | 0.9195509717714485 | 0.975547656030563 | 0.9727330513904973 | 12 | 78: 0.9531 [0.9196, 0.9755] |
| GBM_n128_G_a0.10 | GBM | 128 | 0.1 | 235 | 0.91796875 | 0.8773394692998049 | 0.9485049221701864 | 0.9443751437024567 | 14 | 80: 0.9180 [0.8773, 0.9485] |
| GBM_n256_G_a0.10 | GBM | 256 | 0.1 | 240 | 0.9375 | 0.9004864688995705 | 0.9638572392101775 | 0.9604078888654346 | 16 | 82: 0.9375 [0.9005, 0.9639] |
| GBM_n512_G_a0.10 | GBM | 512 | 0.1 | 231 | 0.90234375 | 0.8592195701659696 | 0.9358003643543249 | 0.9311916202170712 | 18 | 84: 0.9023 [0.8592, 0.9358] |
| GBM_n128_G_a0.05 | GBM | 128 | 0.05 | 247 | 0.96484375 | 0.934313785816128 | 0.9838005891390429 | 0.98153923384891 | 20 | 86: 0.9648 [0.9343, 0.9838] |
| GBM_n256_G_a0.05 | GBM | 256 | 0.05 | 250 | 0.9765625 | 0.9496870965771851 | 0.9913513838038377 | 0.9897448743640003 | 22 | 88: 0.9766 [0.9497, 0.9914] |
| GBM_n512_G_a0.05 | GBM | 512 | 0.05 | 243 | 0.94921875 | 0.9147278740435402 | 0.9726874749946139 | 0.9697047337578512 | 24 | 90: 0.9492 [0.9147, 0.9727] |

The step6b.md column headers (line 66) are "two-sided CP 95%" and "one-sided upper 95%". The L arms are on the even lines in between (step6b.md:69-91; gate.csv odd lines 3-25).

### 5.3 CPU and wall time of the simulations

Labels used below:
- [committed]: a file in git.
- [local, git-ignored]: `results/*/workflow.json`. These are present locally but ignored via `.gitignore:5: results/*`.
- [computed]: a sum I calculated.

**(a) Batch wall-clock, stated (the steps with simulation campaigns):**

| step / block | stated batch wall | source |
|---|---|---|
| 6a: D_train, D_cal, D_test, D_shift (+ 32-pt pilot) | "Batch wall time 16.99 h; elapsed through final publication 23.85 h." | docs/validation/step6b.md:14 [committed] |
| 6a, per phase | pilot 1123.2115468978882 s; D_train 20243.76832016697 s; D_cal 12302.10271274997 s; D_test 17814.635709957918 s; D_shift 9671.45206333301 s; `generation_wall_s` 61155.17035310576; `total_elapsed_s` 85848.8361287117; `workers` 6 | results/step6a/workflow.json:7,15,23,31,39,87,91 [local, git-ignored]; same totals in results/step6b/workflow.json:81,83 |
| 6c: S_nom (900) | `batch_wall_h` 28.2 (includes the about-22 h hung attempt of S_nom_0590 and its 2 h retry: step6c.md:226-230) | docs/validation/step6c.md:13 [committed]; `wall_s` 101640.36873929203, results/step6c/workflow.json:5 [local, git-ignored] |
| 6c: S_dist (240) | `batch_wall_h` 2.31 | docs/validation/step6c.md:14 [committed]; `wall_s` 8330.094386124983, results/step6c/workflow.json:23 [local, git-ignored] |
| 7b: 6 scenario blocks (660 runs) | wall_h 1.549 / 0.08069 / 1.569 / 0.08349 / 1.517 / 0.07235; "Wall time 4.9 h on 6 workers, 2026-10-14 22:58 to 2026-10-15 03:51" | docs/validation/step7b.md:14-19, 256 [committed]; wall_s 5578.155930416891 / 290.46684604207985 / 5649.951094791992 / 300.54940699995495 / 5460.362309667049 / 260.4453179158736, results/step7b/workflow.json:5,23,46,64,87,105 [local, git-ignored] |
| 8a: S_rep (405) | wall_h 3.738 | docs/validation/step8a.md:19 [committed]; `wall_s` 13455.175929083023, results/step8a/workflow.json:20 [local, git-ignored] |
| 6b: surrogate fitting/selection (no simulation) | train 204.75930891698226 s, calibrate 1.7325226671528071 s, diagnostics 1.2646627081558108 s, select 49.03034691698849 s | docs/validation/step6b/provenance.json:60-65 [committed] |

[computed] Sum of the batch wall-clock values in the workflow.json files (6a generation_wall_s + 6c S_nom + S_dist + 7b six phases + 8a S_rep) = 202120.74 s = **56.14 h batch wall-clock**. Per step: 6a 16.99 h, 6c 30.55 h, 7b 4.87 h, 8a 3.74 h. All runs used 6 workers. The total is not stated in any doc.

**(b) Per-point wall time summed (a CPU-time proxy), [computed] from committed `data/generated/*.csv`, column `wall_time_s`:**

| file | rows | sum wall_time_s [s] | [h] |
|---|---|---|---|
| data/generated/D_train.csv | 512 | 126925.89551824937 | 35.2572 |
| data/generated/D_cal.csv | 256 | 71950.9747461211 | 19.9864 |
| data/generated/D_test.csv | 256 | 96183.44238824304 | 26.7176 |
| data/generated/D_shift.csv | 128 | 52479.58074024902 | 14.5777 |
| data/generated/S_nom.csv | 900 | 137029.77293787547 | 38.0638 |
| data/generated/S_dist.csv | 240 | 49279.91703107767 | 13.6889 |
| data/generated/M_ep_inflow_ed_gas.csv | 210 | 33172.7021882867 | 9.2146 |
| data/generated/M_ep_inflow_ed_gas_cold.csv | 10 | 1401.3674892915878 | 0.3893 |
| data/generated/M_thermal_range_low.csv | 210 | 33468.36390975397 | 9.2968 |
| data/generated/M_thermal_range_low_cold.csv | 10 | 1463.5530514167622 | 0.4065 |
| data/generated/M_thermal_range_high.csv | 210 | 32390.43576904456 | 8.9973 |
| data/generated/M_thermal_range_high_cold.csv | 10 | 1339.3647370403633 | 0.3720 |
| data/generated/S_rep.csv | 405 | 80412.76800261531 | 22.3369 |
| **total (3357 runs)** | | **717498.1385092649** | **199.31** |

No NaN values in any file. Caveats, from the docs:
- The sum is per-worker wall time, not measured CPU time. Each worker may use more than one core: "6 requested threads; observed CPU use above 300 %" (docs/preregistration.md:233).
- "Interrupted work after the last saved checkpoint (every 10 cycles) is not included in per-point wall time" (docs/preregistration.md:246-247).
- The largest S_nom value is 7200.17 s (the 2 h limit). The about-22 h hung first attempt of S_nom_0590 (docs/validation/step6c.md:226-230) is therefore not in the per-point sum, though it is in the 28.2 h batch figure.
- The 6a pilot (32 points, 1123.2 s batch) has no separate CSV here.
- No measured CPU time exists for Steps 6a-8a: **CPU time for Steps 6a-8a: NOT FOUND**.

**(c) Steps 2-5 (model development): only per-run figures are stated, no totals.**
- Step 2: "Runtime: 9 s per breakthrough (6000 s) at N = 60" (docs/validation/step2.md:34). Runtime table per case: docs/validation/step2.md:338-350, e.g. final N = 60: mean 8.9 s.
- Step 3: runtime tables with wall_min and cpu_min per case: docs/validation/step3.md:890-968 (one runtime table, rows from :894). "Baseline: 8.9 min CPU (18.1 min wall) for 178 cycles, 2.97 s CPU per cycle" (docs/validation/step3.md:970-971).
- Step 4: "median 5.9, 90th percentile 13.8, max 87.1 min wall (5.4 min CPU median)" (docs/validation/step4.md:311-312; also :41, :60-61, :170-179). Per-run `wall_min`/`cpu_min` columns are in the table header at docs/validation/step4.md:273.
- Step 5: per-run `wall_min` in the capacity-factor table (docs/validation/step5.md:36-45). The step-6 compute plan is an extrapolation, not a measurement (docs/validation/step5.md:77-97).
- **Total CPU/wall for Steps 2-5: NOT FOUND.**

### 5.4 Python and library versions

| item | pyproject.toml [file] | docs/preregistration.md [file] | docs/validation/step6b/provenance.json [file] | installed .venv [venv, `importlib.metadata`] |
|---|---|---|---|---|
| Python | `requires-python = ">=3.10"` (:10) | 3.14.7 (:93; uv CPython 3.14.7 venv :321) | "3.14.7" (:10) | 3.14.7 (CPython, Clang 22.1.3; `.venv/pyvenv.cfg`: uv 0.12.19, cpython-3.14.7-macos-aarch64) |
| numpy | `>=1.24` (:17) | 2.5.3 (:93, :323) | 2.5.3 (:12) | 2.5.3 |
| scipy | `>=1.10` (:18) | 1.18.1 (:93, :323) | 1.18.1 (:13) | 1.18.1 |
| pandas | `>=2.0` (:19) | 3.0.6 (:93, :323) | 3.0.6 (:14) | 3.0.6 |
| pyyaml | `>=6.0` (:20) | – | – | 6.0.3 |
| scikit-learn | `==1.9.1` (:22) | 1.9.1 (:92, :323) | 1.9.1 (:15) | 1.9.1 |
| pymoo | `==0.6.2` (:23) | 0.6.2 (:92, :323) | 0.6.2 (:16) | 0.6.2 |
| statsmodels | `==0.15.0` (:24) | 0.15.0 (:92, :323) | 0.15.0 (:17) | 0.15.0 |
| matplotlib | `>=3.7` (dev extra, :28) | – | – | 3.11.2 |
| pytest | `>=7` (dev extra, :28) | – | – | 9.1.1 |
| joblib (imported: src/h2psa/step6.py:11, scripts/run_step6b.py:25, scripts/run_step8a.py:30) | not listed (sklearn dependency) | – | – | 1.6.0 |
| threadpoolctl (imported: src/h2psa/step6.py:20) | not listed | – | – | 3.7.0 |
| pymupdf (imported only in scripts/digitize_lopes2010.py:20) | not listed | – | – | 1.28.2 |
| lightgbm / xgboost / mapie | not used (no imports in src/, scripts/, tests/; GBM is sklearn `HistGradientBoostingRegressor`, preregistration.md:88; conformal is in-house) | – | – | NOT INSTALLED |

- Lock files: none. There is no uv.lock, requirements*.txt or other freeze file in the working tree or in git. **Lock file: NOT FOUND.**
- docs/preregistration.md:322-323 says the rebuilt venv's packages "were installed from the previous venv's `pip freeze`: all 41 versions are identical", but that freeze is not committed.
- Environment notes: docs/validation/step6c.md:233-235 says "The venv was rebuilt on uv CPython 3.14.7 with the pinned packages (freeze identical)". Machine: Apple M1 Pro, 10 cores (8P + 2E), 16 GB (docs/validation/step5.md:77).

## 6. Figure files and captions

Files in `results/figures/` (git-ignored; regenerated by `python scripts/make_figures.py`), captions from `CAPTIONS` in `scripts/make_figures.py` (also written to `results/figures/captions.md`).

* **F5** — `F5_breakthrough_validation.pdf` / `.png` — caption `scripts/make_figures.py:57-60`:
  > Model validation on the breakthrough runs 1-5 of the compact unit (Lopes 2010, Figs 5.15-5.17): outlet molar flow per species normalised by its feed flow (top) and thermocouple temperatures (bottom); lines: this model (Step 2, final), symbols: digitised experiment; first 3000 s of the 6000-s runs shown (later points lie on the feed plateau).

* **F6** — `F6_psa_cycle_validation.pdf` / `.png` — caption `scripts/make_figures.py:61-63`:
  > PSA cycle validation (experiment cycle, frozen baseline model): pressure at the open column end in cycle 1 and thermocouple temperatures over cycles 1-10 and in cycle 37; lines: model, symbols: measured (Lopes 2010, Figs 5.18-5.19).

* **F7** — `F7_bound_vs_resimulated.pdf` / `.png` — caption `scripts/make_figures.py:64-69`:
  > Upper bound on z (P: point prediction) vs re-simulated impurity z = log10(ppm) of the optimiser-selected points (GP, n_train = 256, alpha = 0.10) for P, G and L (CP: conformal prediction); dashed: 100 ppm limit, dotted: median surrogate mean prediction; filled: violations. (d) Violation rate with Clopper-Pearson 95 % CI at alpha = 0.10, pooled over n_train = 128, 256, 512: per variant (Step 6c, pre-registered), per surrogate (Step 6d, exploratory) and the replication of GP-G, GP-L and GBM-L on new optimiser seeds (Step 8a, pre-registered in Addendum B).

* **FS7** — `FS7_bound_vs_resimulated_n512.pdf` / `.png` — caption `scripts/make_figures.py:70-70`:
  > As Fig. F7 a-c for GP, n_train = 512, alpha = 0.10.

* **F8** — `F8_mechanism.pdf` / `.png` — caption `scripts/make_figures.py:71-72`:
  > Mechanism: (a) GP-L predictive sigma and (b) GBM-L quantile-minus-mean gap Qhat_(1-alpha) - mean at the selected points vs the random D_test points (n_train = 128, 256, 512; alpha = 0.10; Step 6d, exploratory).

* **F9** — `F9_cost_of_guarantee.pdf` / `.png` — caption `scripts/make_figures.py:73-78`:
  > Cost of the guarantee (Step 7a, exploratory, fixed reference point (0, 0)): (a) realised fronts of P, G and L (GP, n_train = 512, alpha = 0.10) and the best-known front; (b) H2 recovery loss of G and L (GP, n_train = 512, alpha = 0.10 and 0.05) vs the oracle-filtered P front at matched productivity (an upper bound on the cost); dotted: the G / L front lies below its lowest realised productivity and its recovery is held at that point's value; left of the vertical line the P front is held likewise.

* **F10** — `F10_robustness.pdf` / `.png` — caption `scripts/make_figures.py:79-82`:
  > Robustness: (a) miscoverage (z > bound) of the G and L arms (n_train = 512, alpha = 0.10) vs adsorbent capacity factor phi on S_dist (Step 6c; binned, Clopper-Pearson 95 % CI) with the RQ5 logistic fit (S_dist intercept, CH4 and CO fractions at their S_dist means); (b) violation rate of the GP n_train = 512 selected points (P at alpha = 0.10; G and L at alpha = 0.10 and 0.05) under the Step 7b model-form scenarios (descriptive).

Other file: `captions.md`. All files present: F10_robustness.pdf, F10_robustness.png, F5_breakthrough_validation.pdf, F5_breakthrough_validation.png, F6_psa_cycle_validation.pdf, F6_psa_cycle_validation.png, F7_bound_vs_resimulated.pdf, F7_bound_vs_resimulated.png, F8_mechanism.pdf, F8_mechanism.png, F9_cost_of_guarantee.pdf, F9_cost_of_guarantee.png, FS7_bound_vs_resimulated_n512.pdf, FS7_bound_vs_resimulated_n512.png, captions.md.

## 7. Thesis page check (Lopes 2010, p. 256)

Searched docs/**/*.md, docs/preregistration.md, data/PROVENANCE.md, data/params/*.yaml, data/benchmarks/ (README and CSVs) and CLAUDE.md:
- "p.256" is found.
- "p. 256", "page 256" and "PDF page 290" are not found.

**Verbatim quote (present, copied from the thesis):**

docs/validation/step3.md:31-36:
> * p.256: "it was concluded that a hydrogen recovery of 81.6% was obtained in the PSA experiment with a H2 purity
>   of 99.98054% contaminated with 63 ppm of CO (simulated values) and a process productivity of 101.1 molH2
>   kgads-1 day-1." p.241 (chapter 6) states the practice: "Due to the small number of experimental points, report
>   experimental values of these performance parameters may not be very accurate. Since the simulation predicts with
>   good accuracy the experimental results obtained, the performance parameters based on simulated results were
>   reported."

The same quote appears again at docs/validation/step3.md:411-416, in bold, with "The experiment's purity, CO and recovery used in gate 2 are therefore the author's simulation of the experiment, not measurements."

Note: the thesis gives purity as 99.98054 % (not 99.98 %). Elsewhere the repo rounds it to 99.9805 % (data/benchmarks/lopes2010_psa_experiment.csv:2; docs/validation/step2.md:135; docs/validation/step3.md:716).

**Paraphrases and short quotes:**
- data/PROVENANCE.md:126: `| PSA experiment: purity 99.98054 %, CO 63 ppm, recovery 81.6 %, productivity 101.1 mol/kg/day | **simulated** (author's model of the experiment) | p.256 "(simulated values)"; practice stated on p.241 |`
- data/PROVENANCE.md:51-52: "... the author's simulation of the experiment (p.256), never against measured data"
- data/benchmarks/README.md:3-5: "The purity, CO, recovery and productivity are the author's SIMULATED values for the experiment (p.256: "(simulated values)"; practice stated on p.241), not measurements."
- docs/validation/step3.md:37-40: "The PSA experiment's purity, CO, recovery and productivity are the author's simulated values. ... treated as the author's 18th simulation (verification, together with Table 5.10)."
- docs/validation/step3.md:83: "Simulation 18 is the experiment cycle (author's simulated values, p.256)."
- docs/validation/step3.md:606-607, :616: decision text citing "(p.256)".
- CLAUDE.md:35: "are the author's SIMULATED values (p.256, practice stated on p.241)".
- p.256 is not cited in data/params/*.yaml or docs/preregistration.md. The only hit in data/params is compact_column_lopes2010.yaml:29, which is about Fig. 5.20a p.197, not p.256.

Page mapping (per CLAUDE.md, PDF = printed + 34): printed p.256 is PDF p.290. No repo file states "PDF page 290". This was not checked against the PDF (literature/ was not opened).
