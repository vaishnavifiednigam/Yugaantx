#!/usr/bin/env python3
"""Surgery part 3: (a) strip hero reproducibility line + chips row, (b) Study shelf —
every completed run auto-saved with its graphs, kept until user removes; pin & compare."""
src = open("app.py").read()
n = 0

def sub(old: str, new: str, cnt: int = 1):
    global src, n
    c = src.count(old)
    assert c == cnt, f"anchor count {c} != {cnt} for: {old[:90]!r}"
    src = src.replace(old, new)
    n += 1

# ── (a) hero cleanup ────────────────────────────────────────────────────────
sub('''        "<div class='yu-sub'>A small synthetic society of rule-based agents — they gather, trade, share, co-operate "
        "and steal in a seeded world. Every score, chart and card on this page is computed <b>live from that "
        "simulation</b>: same seed + same settings ⇒ the exact same society, every time.</div>"
        "<div style='margin-top:2px;'>"
        + chip("⚠ SIMULATED DATA — not real-world", "#3b1224", "#fda4af") + " "
        + chip("single file · no internet · optional on-device learning", "#12233d", "#93c5fd") + " "
        + chip("100% reproducible from the seed", "#0f2b26", "#5eead4") +
        "</div></div>", unsafe_allow_html=True)''',
    '''        "<div class='yu-sub'>A small synthetic society of agents — they gather, trade, share, co-operate "
        "and steal in a seeded world. Every score and chart on this page is computed <b>live from that "
        "simulation</b>.</div>"
        "</div>", unsafe_allow_html=True)''')

# ── (b1) shelf core + callbacks (inserted before after_run) ─────────────────
sub('''def after_run(res: "SimResult") -> None:
    """Session-only group memory: store the elites' learned bias when carry is on (RAM only,
    disappears with the tab — nothing is saved online)."""
    bo = res.meta.get("brain_out")
    if bo is None:
        return''',
    '''SHELF_MAX = 3          # unpinned snapshots kept automatically
SHELF_PIN_MAX = 3      # user-pinned snapshots kept until removed


def _shelf_find(key: str) -> Optional[dict]:
    for e in st.session_state.get("shelf", []):
        if e["key"] == key:
            return e
    return None


def _shelf_remove(key: str) -> None:
    st.session_state["shelf"] = [e for e in st.session_state.get("shelf", []) if e["key"] != key]


def _shelf_pin(key: str) -> None:
    e = _shelf_find(key)
    if not e:
        return
    if not e["pinned"] and sum(1 for x in st.session_state["shelf"] if x["pinned"]) >= SHELF_PIN_MAX:
        st.session_state["shelf_note"] = (f"⚠ already {SHELF_PIN_MAX} pinned — unpin one first (pins are kept "
                                          "forever on purpose, so they are capped).")
        return
    e["pinned"] = not e["pinned"]


def _shelf_load(key: str) -> None:
    e = _shelf_find(key)
    if not e:
        return
    st.session_state["res"] = e["res"]
    st.session_state.pop("frame_slider", None)
    st.session_state["insp_sel"] = None
    st.session_state["last_run_msg"] = (f"🗂 loaded saved run — {e['label']} · snapshotted at {e['ts']}. "
                                        "Nothing was re-simulated: these are the exact figures from that finished "
                                        "run, restored into every tab.")


def shelf_add(res: "SimResult") -> None:
    """Auto-save every completed run for study; the oldest unpinned snapshot is dropped past SHELF_MAX."""
    cfg = res.cfg
    rd = res.rounds_df
    deaths = int(rd["deaths"].iloc[-1]) if len(rd) else 0
    label = (f"seed {cfg.seed} · {cfg.population}×{cfg.rounds} r · "
             + ("🧠 learned" if cfg.policy == "LEARNED" else "📜 rules")
             + (f" · gen {int(res.meta.get('brain_gen', 0)) + 1}" if cfg.policy_carry else "")
             + f" · {res.meta['alive_end']}/{res.n0} survived · {deaths} deaths")
    shelf = st.session_state.setdefault("shelf", [])
    shelf.append({"key": f"s{int(time.time_ns() % 10**9)}", "label": label,
                  "ts": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                  "res": res, "pinned": False})
    while sum(1 for e in shelf if not e["pinned"]) > SHELF_MAX:
        first = next(i for i, e in enumerate(shelf) if not e["pinned"])
        shelf.pop(first)


def after_run(res: "SimResult") -> None:
    """Runs finished from any path land on the study shelf; carry-memory bookkeeping too."""
    shelf_add(res)
    bo = res.meta.get("brain_out")
    if bo is None:
        return''')

# ── (b2) render_shelf (insert before the 🧠 learning lab section) ───────────
sub('''
# ── 🧠 learning lab ─────────────────────────────────────────────────────────
def render_learning(res: SimResult) -> None:''',
    '''
# ── 🗂 study shelf — completed runs kept until YOU remove them ──────────────
def _shelf_charts(rd: pd.DataFrame, title: str, h: int = 250) -> go.Figure:
    series = [("cooperation_rate", "co-operation", PALETTE["green"], None),
              ("competition_rate", "competition", PALETTE["red"], None),
              ("trade_rate", "trade", PALETTE["violet"], "dot"),
              ("avg_trust", "mean trust", PALETTE["teal"], "dash")]
    f = multi_line(rd, series, title, h=h)
    f.update_legend = None  # noqa (keeps linters from removing the call below)
    f.update_layout(legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9.5)))
    return f


def _shelf_stats_md(r: "SimResult") -> str:
    rd = r.rounds_df
    def eol(col: str) -> tuple[float, float]:
        return float(rd[col].iloc[0]), float(rd[col].iloc[-1])
    rows = [("co-operation", "cooperation_rate", "%"), ("competition", "competition_rate", "%"),
            ("mean trust", "avg_trust", ""), ("survival", "survival", "%"),
            ("wealth Gini", "gini_wealth", ""), ("reward / agent / r", "avg_reward", "")]
    lines = []
    for lab, col, unit in rows:
        a, b = eol(col)
        if unit == "%":
            a, b = a * 100, b * 100
        arr = "▲" if b > a + 1e-9 else ("▼" if b < a - 1e-9 else "→")
        lines.append(f"<div style='display:flex;justify-content:space-between;font-size:12.5px;'>"
                     f"<span style='color:#9fb3d1'>{lab}</span>"
                     f"<span style='color:#e6edf7'><b>{a:.1f}</b> → <b>{b:.1f}</b>{unit} {arr}</span></div>")
    if "policy_entropy" in rd.columns:
        e0, e1 = eol("policy_entropy")
        d0, d1 = eol("divergence_rate")
        lines.append(f"<div style='display:flex;justify-content:space-between;font-size:12.5px;'>"
                     f"<span style='color:#a78bfa'>policy entropy</span>"
                     f"<span style='color:#e6edf7'><b>{e0:.2f}</b> → <b>{e1:.2f}</b> bits</span></div>")
        lines.append(f"<div style='display:flex;justify-content:space-between;font-size:12.5px;'>"
                     f"<span style='color:#a78bfa'>off-rule decisions</span>"
                     f"<span style='color:#e6edf7'>avg <b>{d0*0+d1*0+float(rd['divergence_rate'].mean())*100:.0f}%</b></span></div>")
    return ("<div class='yu-card' style='padding:10px 12px'>"
            "<div style='color:#7c8db0;font-size:11px;letter-spacing:.08em;margin-bottom:6px;'>FIRST ROUND → LAST ROUND</div>"
            + "".join(lines) + "</div>")


def render_shelf() -> None:
    shelf = st.session_state.get("shelf", [])
    if not shelf:
        st.markdown("<div class='empty-state'><div style='font-size:34px'>🗂</div>"
                    "<h3>Nothing saved yet</h3><p style='max-width:640px;margin:0 auto'>Every completed run lands "
                    "here automatically — full graphs and stats, frozen exactly as they finished. Snapshots stay "
                    f"on the shelf until YOU press ✖ (last {SHELF_MAX} unpinned are kept; 📌 pins are never "
                    "auto-removed). Press Start in the sidebar to make the first one.</p></div>",
                    unsafe_allow_html=True)
        return
    note = st.session_state.pop("shelf_note", None)
    if note:
        st.warning(note)
    st.caption(f"🗂 Study shelf — frozen copies of finished runs, held in this browser session's memory only "
               f"(temporary — download if you want to keep them). Unpinned shelf size: {SHELF_MAX} · "
               f"pinned are kept until you press ✖.")
    for idx, e in enumerate(list(shelf)):
        r = e["res"]
        head = ("📌 " if e["pinned"] else "") + f"{e['ts']} · {e['label']}"
        with st.expander(head, expanded=(idx == len(shelf) - 1)):
            rd = r.rounds_df
            cA, cB = st.columns([2.6, 1])
            with cA:
                st.plotly_chart(_shelf_charts(rd, f"how this run unfolded — {e['label']}"), width="stretch")
            with cB:
                st.markdown(_shelf_stats_md(r), unsafe_allow_html=True)
            if "policy_entropy" in rd.columns:
                st.plotly_chart(multi_line(rd, [
                    ("policy_entropy", "policy entropy (bits)", PALETTE["violet"], None),
                    ("divergence_rate", "off-rule share", "#a78bfa", "dot"),
                ], "🧠 training story of this run", h=190), width="stretch")
            b1, b2, b3, b4 = st.columns([1.15, 0.8, 0.8, 2.4])
            b1.button("↩ Load into tabs", width="stretch", key=f"sh_load_{e['key']}", on_click=_shelf_load,
                      args=(e["key"],), help="restores this exact snapshot as the current run — no re-simulation")
            b2.button("📌 Pin" if not e["pinned"] else "🧵 Unpin", width="stretch", key=f"sh_pin_{e['key']}",
                      on_click=_shelf_pin, args=(e["key"],))
            b3.button("✖ Remove", width="stretch", key=f"sh_rm_{e['key']}", on_click=_shelf_remove, args=(e["key"],))
            b4.caption("Loading fills Society view, Trends, Learning lab, Inspector and Compare with this "
                       "run's real values — scroll, scrub, inspect, export at leisure.")
            st.download_button("⬇ this run's per-round CSV", df_to_csv(rd),
                               f"yudaant_shelf_{e['key']}.csv", "text/csv", key=f"sh_csv_{e['key']}")
    if len(shelf) >= 2:
        st.divider()
        st.markdown("**⚖ Compare any two saved runs** — overlay their recorded curves, side by side")
        opts = {f"#{i+1} · {e['ts']} · {e['label']}": i for i, e in enumerate(shelf)}
        c1, c2 = st.columns(2)
        ka = c1.selectbox("run A", list(opts), index=len(shelf) - 1, key="sh_cmp_a")
        kb = c2.selectbox("run B", list(opts), index=max(0, len(shelf) - 2), key="sh_cmp_b")
        i_a, i_b = opts[ka], opts[kb]
        if i_a == i_b:
            st.info("Pick two different snapshots to overlay them.")
            return
        da = shelf[i_a]["res"].rounds_df
        db = shelf[i_b]["res"].rounds_df
        fig = go.Figure()
        for df, nm, dash in ((da, ka.split(" · ")[1], "solid"), (db, kb.split(" · ")[1], "dot")):
            for col, lab, colr in (("cooperation_rate", "co-op", PALETTE["green"]),
                                   ("competition_rate", "compete", PALETTE["red"]),
                                   ("avg_trust", "trust", PALETTE["teal"]),
                                   ("survival", "survival", PALETTE["blue"])):
                if col not in df.columns:
                    continue
                fig.add_trace(go.Scatter(x=df["round"], y=df[col], name=f"{nm} · {lab}",
                                         line=dict(width=2.1 if dash == "solid" else 1.5, color=colr, dash=dash),
                                         hovertemplate=f"r%{{x}} · {nm} {lab}: %{{y:.3f}}<extra></extra>"))
        style_fig(fig, 300)
        fig.update_layout(title=dict(text="solid = run A · dotted = run B (raw recorded values, no smoothing)",
                                     font=dict(size=12.5, color="#c9d7ee")), xaxis_title="round")
        st.plotly_chart(fig, width="stretch")
        rows = []
        for col, lab in (("cooperation_rate", "co-operation rate"), ("competition_rate", "competition rate"),
                         ("avg_trust", "mean trust"), ("survival", "survival"), ("gini_wealth", "wealth Gini"),
                         ("avg_reward", "reward / agent / round")):
            va = float(da[col].iloc[-1]); vb = float(db[col].iloc[-1])
            rows.append({"metric": lab, "A (end)": round(va, 3), "B (end)": round(vb, 3), "Δ B−A": round(vb - va, 3)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Two independent samples of the model — differences describe these two runs, not “laws”.")


# ── 🧠 learning lab ─────────────────────────────────────────────────────────
def render_learning(res: SimResult) -> None:''')

# ── (b3) tab wiring: new 🗂 Study shelf at index 3 ──────────────────────────
sub('''    tabs = st.tabs(["🌍 Society view", "📈 Trends & graphs", "🧠 Learning lab", "🔎 Agent inspector",
                     "⚖ Compare A/B", "🧭 Discoveries", "💾 Data & export", "📖 Method & limits"])''',
    '''    tabs = st.tabs(["🌍 Society view", "📈 Trends & graphs", "🧠 Learning lab", "🗂 Study shelf",
                     "🔎 Agent inspector", "⚖ Compare A/B", "🧭 Discoveries", "💾 Data & export",
                     "📖 Method & limits"])''')
sub('''    _tab_body(2, render_learning, "🧠 learning curves appear after a run with policy = LEARNED")
    _tab_body(3, render_inspector, "🔎 every agent's decision story appears here")
    with tabs[4]:
        render_compare()
    _tab_body(5, render_discoveries, "🧭 automatic findings appear here")
    _tab_body(6, render_data, "💾 CSV / JSON exports appear here")
    with tabs[7]:
        render_method(res, checks)''',
    '''    _tab_body(2, render_learning, "🧠 learning curves appear after a run with policy = LEARNED")
    with tabs[3]:
        render_live_wait() if live else render_shelf()
    _tab_body(4, render_inspector, "🔎 every agent's decision story appears here")
    with tabs[5]:
        render_compare()
    _tab_body(6, render_discoveries, "🧭 automatic findings appear here")
    _tab_body(7, render_data, "💾 CSV / JSON exports appear here")
    with tabs[8]:
        render_method(res, checks)''')

print(f"patched {n} sites part-3")
open("app.py", "w").write(src)
