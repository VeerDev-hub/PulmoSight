"""
PulmoSight — Streamlit app.

    streamlit run streamlit_app.py

Pages: Introduction · Model comparison · Test an X-ray
Research / portfolio project only. Not a medical device and not a diagnosis.
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from pulmosight.inference import MODEL_INFO, Screener, load_ensemble, load_results

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "assets" / "samples"
TRAILER = ROOT / "assets" / "trailer" / "PulmoSight_trailer.mp4"
COLORS = {"ensemble": "#D97706", "efficientnet_b0": "#059669", "pulmonet": "#7C3AED", "simple_cnn": "#64748B"}
SHORT = {key: name for key, (name, _) in MODEL_INFO.items()}

st.set_page_config(page_title="PulmoSight · TB detection from chest X-rays", page_icon="🫁", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 4.6rem; max-width: 1180px; }
  .eyebrow { color:#087f78; font-size:.75rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.2rem; }
  .hero h1 { font-size:2.7rem; line-height:1.1; margin:.2rem 0 .8rem; }
  .lede { color:#55625e; font-size:1.12rem; line-height:1.65; max-width:760px; }
  .card { background:#fff; border:1px solid #dce3dc; border-radius:14px; padding:1.1rem 1.2rem; height:100%; }
  .card h4 { margin:.1rem 0 .4rem; font-size:1.05rem; } .card p { color:#55625e; margin:0; font-size:.93rem; line-height:1.55; }
  .step { font-weight:700; color:#087f78; font-size:.8rem; letter-spacing:.08em; }
  .verdict { border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:1rem; border:1px solid; }
  .verdict.tb { background:#fdf0ec; border-color:#efc3b8; } .verdict.normal { background:#ebf6f2; border-color:#b9dacc; }
  .verdict .label { font-size:.78rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#55625e; }
  .verdict .value { font-size:2.1rem; font-weight:700; line-height:1.15; }
  .verdict.tb .value { color:#a63d2a; } .verdict.normal .value { color:#075f5a; }
  .verdict .sub { color:#55625e; font-size:.95rem; margin-top:.2rem; }
  .meter { position:relative; height:14px; background:#e7ece7; border-radius:99px; margin:.6rem 0 .35rem; }
  .meter.small { height:9px; margin:.35rem 0 .2rem; }
  .meter .fill { position:absolute; left:0; top:0; bottom:0; border-radius:99px; }
  .meter .tick { position:absolute; top:-6px; bottom:-6px; width:3px; margin-left:-1.5px; background:#17211f; border-radius:2px; }
  .meter-labels { display:flex; justify-content:space-between; color:#6b7773; font-size:.8rem; }
  .vote { display:flex; justify-content:space-between; font-size:.9rem; margin-top:.55rem; }
  .vote b.tb { color:#a63d2a; } .vote b.normal { color:#075f5a; }
  .facts { display:grid; grid-template-columns:1fr 1fr; gap:.6rem; margin:.2rem 0 .9rem; }
  .facts div { background:#fff; border:1px solid #dce3dc; border-radius:10px; padding:.6rem .8rem; }
  .facts span { display:block; color:#6b7773; font-size:.8rem; } .facts b { font-size:1.45rem; line-height:1.3; }
  .truth { font-size:.95rem; padding:.55rem .8rem; background:#f2f5f1; border-radius:8px; display:inline-block; margin-bottom:.8rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the trained models…")
def get_screener():
    return Screener()


@st.cache_data
def get_results():
    return load_results(), load_ensemble()


@st.cache_data
def get_samples():
    path = SAMPLES / "samples.json"
    return json.loads(path.read_text()) if path.exists() else []


def model_rows():
    """One row per model (3 networks + the ensemble) with test metrics and cost, ranked by test accuracy."""
    results, ens = get_results()
    rows = []
    for label, m in results["models"].items():
        b, e = m["metrics"]["balanced"], m["efficiency"]
        rows.append({"key": m["arch"], "Model": label, "metrics": b, "confusion": m["confusion_balanced"], "roc": m["roc"],
                     "Parameters (M)": e["Parameters (M)"], "GFLOPs": e["GFLOPs / image"], "CPU latency (ms)": e["CPU latency (ms/img)"],
                     "Training (min)": e["Training time (min)"], "Epochs": str(int(e["Epochs"]))})
    nets = list(rows)
    rows.append({"key": "ensemble", "Model": "Ensemble (all three)", "metrics": ens["test_metrics"], "confusion": ens["confusion_balanced"],
                 "roc": ens["roc"], "Parameters (M)": sum(r["Parameters (M)"] for r in nets), "GFLOPs": sum(r["GFLOPs"] for r in nets),
                 "CPU latency (ms)": sum(r["CPU latency (ms)"] for r in nets), "Training (min)": sum(r["Training (min)"] for r in nets),
                 "Epochs": "—"})
    for r in rows:
        b = r["metrics"]
        r.update({"Short": SHORT[r["key"]], "Accuracy": b["Accuracy"][0], "acc_lo": b["Accuracy"][1], "acc_hi": b["Accuracy"][2],
                  "AUROC": b["AUROC"][0], "auc_lo": b["AUROC"][1], "auc_hi": b["AUROC"][2],
                  "Sensitivity": b["Sensitivity"][0], "Specificity": b["Specificity"][0]})
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df["Accuracy"].round(3).rank(ascending=False, method="min").astype(int))
    return df


# ----------------------------------------------------------------------------------------------------- Introduction
def page_intro():
    results, ens = get_results()
    df = model_rows()
    best = df.iloc[0]
    total = results["train_size"] + results["val_size"] + results["test_size"]

    st.markdown('<div class="hero"><div class="eyebrow">Deep learning · chest X-ray screening</div>'
                '<h1>PulmoSight: detecting tuberculosis in chest X-rays</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lede">Tuberculosis is one of the world\'s deadliest infectious diseases, and a chest X-ray is often the '
        'first screening step. <b>PulmoSight</b> is a deep-learning system that looks at a frontal chest X-ray and estimates '
        'whether it shows signs of TB, so that suspicious cases can be prioritised for confirmatory testing.</p>',
        unsafe_allow_html=True)
    if TRAILER.exists():
        st.video(str(TRAILER), autoplay=True, muted=True, loop=True)
    st.write("")

    c = st.columns(4)
    c[0].metric("Test accuracy", f"{best['Accuracy']:.1%}", help=f"{best['Model']} on {results['test_size']:,} X-rays it never saw")
    c[1].metric("AUROC", f"{best['AUROC']:.3f}", help="1.0 = perfect separation of TB and Normal, 0.5 = chance")
    c[2].metric("Chest X-rays used", f"{total:,}", help=f"{results['train_size']:,} train · {results['val_size']:,} validation · {results['test_size']:,} test")
    c[3].metric("Models", "3 + ensemble", help="Simple CNN, PulmoNet (custom CNN), EfficientNet-B0 (transfer learning), and their weighted ensemble")

    st.subheader("How it works")
    steps = [("01 · Upload", "A frontal chest X-ray in PNG, JPG, BMP or TIF format."),
             ("02 · Preprocess", "Converted to grayscale and resized to 512×512, exactly as during training."),
             ("03 · Classify", "Three convolutional neural networks each score the image for TB vs Normal."),
             ("04 · Decide", "Their calibrated probabilities are combined and compared with a threshold chosen on validation data.")]
    for col, (title, text) in zip(st.columns(4), steps):
        col.markdown(f'<div class="card"><div class="step">{title}</div><p>{text}</p></div>', unsafe_allow_html=True)

    st.subheader("The models")
    for col, key in zip(st.columns(4), ["simple_cnn", "pulmonet", "efficientnet_b0", "ensemble"]):
        row = df[df.key == key].iloc[0]
        col.markdown(f'<div class="card" style="border-top:4px solid {COLORS[key]}"><h4>{SHORT[key]}</h4>'
                     f'<p>{MODEL_INFO[key][1]}</p><p style="margin-top:.6rem"><b>{row.Accuracy:.1%}</b> test accuracy</p></div>',
                     unsafe_allow_html=True)

    st.subheader("What it can and cannot do")
    a, b = st.columns(2)
    a.markdown("""
**It can**
- Flag frontal chest X-rays that look consistent with TB
- Give a calibrated probability, not just a yes/no
- Show how each of the three networks voted
- Run in under a second on an ordinary CPU
""")
    b.markdown("""
**It cannot**
- Diagnose TB: that needs sputum, culture or molecular tests
- Show where the disease is, or how severe it is
- Guarantee the same accuracy on X-rays from other hospitals or scanners
""")
    st.write("")
    st.page_link(PAGES["test"], label="Try it on an X-ray →", icon="🩻")


# ------------------------------------------------------------------------------------------------- Model comparison
def page_compare():
    results, ens = get_results()
    df = model_rows()
    order = [SHORT[k] for k in df.key]
    color_scale = alt.Scale(domain=[SHORT[k] for k in COLORS], range=list(COLORS.values()))

    st.markdown(f'<div class="eyebrow">Held-out test set · {results["test_size"]:,} X-rays</div>', unsafe_allow_html=True)
    st.title("Model comparison")
    st.markdown('<p class="lede">Three networks trained on the same images with the same augmentation and selection rule: '
                'a basic CNN, a custom CNN built for this project, and a transfer-learning model, plus an ensemble that combines '
                'them. Every weight, threshold and calibration was fixed on the validation set before the test set was opened.</p>',
                unsafe_allow_html=True)

    for col, (_, r) in zip(st.columns(4), df.iterrows()):
        tied = (df.Rank == r.Rank).sum() > 1
        col.markdown(f'<div class="card" style="border-top:4px solid {COLORS[r.key]}">'
                     f'<div class="step" style="color:{COLORS[r.key]}">#{r.Rank}{" · tied" if tied else ""}</div>'
                     f'<h4>{r.Model}</h4><div style="font-size:2.2rem;font-weight:700;line-height:1.2">{r.Accuracy:.1%}</div>'
                     f'<p>test accuracy (95% CI {r.acc_lo:.1%}–{r.acc_hi:.1%})</p>'
                     f'<p style="margin-top:.5rem">AUROC <b>{r.AUROC:.3f}</b><br>sensitivity <b>{r.Sensitivity:.1%}</b> · '
                     f'specificity <b>{r.Specificity:.1%}</b></p></div>', unsafe_allow_html=True)
    st.write("")

    st.subheader("Accuracy and AUROC with 95% confidence intervals")
    chart_df = df[["Short", "Model", "Accuracy", "acc_lo", "acc_hi", "AUROC", "auc_lo", "auc_hi"]]
    for col, (metric, lo, hi, fmt) in zip(st.columns(2), [("Accuracy", "acc_lo", "acc_hi", ".1%"), ("AUROC", "auc_lo", "auc_hi", ".3f")]):
        base = alt.Chart(chart_df).encode(y=alt.Y("Short:N", sort=order, title=None))
        bars = base.mark_bar(size=24, cornerRadiusEnd=4).encode(
            x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[0.75, 1.0]), axis=alt.Axis(format=fmt, tickCount=6), title=None),
            color=alt.Color("Short:N", scale=color_scale, legend=None), tooltip=["Model", alt.Tooltip(f"{metric}:Q", format=fmt)])
        err = base.mark_rule(strokeWidth=2, color="#17211f").encode(x=f"{lo}:Q", x2=f"{hi}:Q")
        text = base.mark_text(align="left", dx=6, fontWeight="bold").encode(x=f"{hi}:Q", text=alt.Text(f"{metric}:Q", format=fmt))
        col.markdown(f"**Test {metric.lower() if metric == 'Accuracy' else metric}**")
        col.altair_chart((bars + err + text).properties(height=230), width="stretch")

    a, b = st.columns([1.05, 1])
    with a:
        st.subheader("ROC curves")
        roc = pd.DataFrame([{"Model": r.Short, "False-positive rate": p[0], "Sensitivity": p[1]} for _, r in df.iterrows() for p in r.roc])
        line = alt.Chart(roc).mark_line(strokeWidth=2.2).encode(
            x=alt.X("False-positive rate:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Sensitivity:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=color_scale, legend=alt.Legend(orient="bottom-right", title=None)))
        diag = alt.Chart(pd.DataFrame({"False-positive rate": [0, 1], "Sensitivity": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="#9aa8a2").encode(x="False-positive rate:Q", y="Sensitivity:Q")
        st.altair_chart((diag + line).properties(height=380), width="stretch")
    with b:
        st.subheader("Errors on the test set")
        st.caption("At each model's balanced threshold. Missed = TB predicted as Normal. False alarm = Normal predicted as TB.")
        for _, r in df.iterrows():
            (tn, fp), (fn, tp) = r.confusion
            c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
            c1.markdown(f"**{r.Short}**")
            c2.metric("Correct", f"{tn + tp:,}")
            c3.metric("Missed TB", f"{fn:,}")
            c4.metric("False alarms", f"{fp:,}")

    st.subheader("Cost: size, compute and training time")
    cost = df[["Model", "Parameters (M)", "GFLOPs", "CPU latency (ms)", "Training (min)", "Epochs"]].set_index("Model")
    st.dataframe(cost.style.format({"Parameters (M)": "{:.2f}", "GFLOPs": "{:.2f}", "CPU latency (ms)": "{:.0f}", "Training (min)": "{:.0f}"}),
                 width="stretch")
    st.caption("The ensemble runs all three networks, so its cost is the sum of theirs.")

    st.subheader("Are the differences real?")
    v = ens["vs_effnet"]
    sig = pd.DataFrame(results["significance"] + [{"comparison": "Ensemble  vs  EfficientNet-B0 (transfer)", "ΔAccuracy": v["delta_accuracy"],
                                                   "ΔAUROC": v["delta_auroc"], "DeLong p": v["delong_p"], "McNemar p": v["mcnemar_p"]}])
    sig["Verdict"] = ["significant" if (r["DeLong p"] < 0.05 or r["McNemar p"] < 0.05) else "not significant" for _, r in sig.iterrows()]
    sig = sig.rename(columns={"comparison": "Comparison"})[["Comparison", "ΔAccuracy", "ΔAUROC", "DeLong p", "McNemar p", "Verdict"]]
    st.dataframe(sig.style.format({"ΔAccuracy": "{:+.1%}", "ΔAUROC": "{:+.3f}",
                                   "DeLong p": lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}",
                                   "McNemar p": lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("DeLong compares AUROC; McNemar compares which images each model gets wrong. p < 0.05 means the gap is unlikely to be chance.")

    st.subheader("What this shows")
    st.markdown(f"""
1. **A basic CNN gets surprisingly far.** Five plain convolution blocks (0.39 M parameters) tie our custom PulmoNet on test accuracy,
   and the AUROC difference is not significant.
2. **Architecture tricks alone did not pay off.** PulmoNet's residual blocks and attention use 10× the parameters and 4× the compute
   without a measurable gain.
3. **Transfer learning is the strongest single model.** ImageNet pre-training gives EfficientNet-B0 a significant lead over both
   from-scratch networks, reached in the fewest epochs.
4. **Together they are better still.** A weighted vote of all three ({ens['test_metrics']['Accuracy'][0]:.1%}) significantly beats
   EfficientNet-B0 alone: it gets {v['only_ensemble_right']} test images right that EfficientNet misses, against {v['only_effnet_right']} the
   other way. It mainly cuts false alarms; the trade-off is slightly lower sensitivity.
""")
    with st.expander("Limitations found during evaluation"):
        st.markdown("""
- **Shortcut learning.** With the lungs masked out, the models can still partly tell TB from Normal, so they also use cues outside the
  lungs (film borders, side markers, scanner differences). This is why the app does not show heatmaps: they often highlight bones
  or markers, which would be misleading as an explanation of TB.
- **Image quality.** Strong noise, blur or very low resolution reduce accuracy, especially for EfficientNet-B0.
- **One data source.** All test images come from the same collection as the training data; accuracy on other hospitals' X-rays is unknown.
- **A robustness experiment** (lung-cropped retraining, notebooks 08–10) made the models far more robust to noise but cost 2–4
  accuracy points, so the app uses the more accurate original models.
""")


# ---------------------------------------------------------------------------------------------------- Test an X-ray
def page_test():
    screener = get_screener()
    samples = get_samples()
    st.markdown('<div class="eyebrow">Try the model</div>', unsafe_allow_html=True)
    st.title("Test an X-ray")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        options = list(MODEL_INFO)
        model = st.radio("1 · Choose a model", options, index=0, key="model",
                         format_func=lambda k: f"{SHORT[k]}{'  (recommended)' if k == 'ensemble' else ''}",
                         captions=[f"{screener.models[k]['test']['Accuracy'][0]:.1%} test accuracy · {MODEL_INFO[k][1]}" for k in options])

        st.markdown("**2 · Upload a chest X-ray**")
        upload = st.file_uploader("Frontal chest X-ray", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], label_visibility="collapsed")
        st.caption("Images are processed in memory and not stored.")

        if samples:
            st.markdown("**…or pick a sample from the test set**")
            cols = st.columns(len(samples))
            for i, (col, s) in enumerate(zip(cols, samples)):
                col.image(str(SAMPLES / s["thumb"]), width="stretch")
                if col.button(f"#{i + 1}", key=f"sample_{i}", width="stretch"):
                    st.session_state["sample"] = i
            if upload is not None:
                st.session_state.pop("sample", None)

    with right:
        source, truth = None, None
        if upload is not None:
            source, name = upload, upload.name
        elif "sample" in st.session_state:
            s = samples[st.session_state["sample"]]
            source, name, truth = SAMPLES / s["file"], f"Sample #{st.session_state['sample'] + 1} (test set)", s["truth"]

        if source is None:
            st.info("Choose a model, then upload an X-ray or click a sample to see the result here.")
            return
        try:
            r = screener.predict(source, model)
        except Exception as exc:
            st.error(f"Could not read this image ({exc.__class__.__name__}). Please upload a PNG, JPG, BMP or TIF chest X-ray.")
            return
        show_result(r, name, truth)


def meter(p, thr, color, small=False):
    return (f'<div class="meter{" small" if small else ""}"><div class="fill" style="width:{p * 100:.1f}%;background:{color}"></div>'
            f'<div class="tick" style="left:{thr * 100:.1f}%"></div></div>')


def show_result(r, name, truth):
    results, _ = get_results()
    is_tb = r["prediction"] == "TB"
    who = "The ensemble" if r["model_key"] == "ensemble" else r["model"]
    title = "TB suspected" if is_tb else "No TB detected"
    sub = f"{who} finds this X-ray consistent with tuberculosis." if is_tb else f"{who} finds no pattern consistent with tuberculosis."
    st.markdown(f'<div class="verdict {"tb" if is_tb else "normal"}"><div class="label">Result · {r["model"]}</div>'
                f'<div class="value">{title}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)
    if truth:
        correct = (truth == "TB") == is_tb
        st.markdown(f'<div class="truth">Ground truth for this sample: <b>{truth}</b> · the model is '
                    f'<b>{"correct" if correct else "wrong"}</b> here</div>', unsafe_allow_html=True)
    for w in r["warnings"]:
        st.warning(w)

    img, info = st.columns([1, 1.15])
    img.image(r["display"], caption=name, width="stretch")
    with info:
        st.markdown(f'**TB probability: {r["p_tb"]:.1%}**' + meter(r["p_tb"], r["threshold"], "#bd4f3b" if is_tb else "#087f78") +
                    f'<div class="meter-labels"><span>0%</span><span>threshold {r["threshold"]:.0%}</span><span>100%</span></div>',
                    unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="facts"><div><span>Decision</span><b>{r["prediction"]}</b></div>'
                    f'<div title="How far the probability moved from the threshold towards 0% or 100%: Clear = at least 60% of the way, '
                    f'Moderate = at least 25%, otherwise Borderline."><span>Certainty</span><b>{r["confidence"]}</b></div></div>',
                    unsafe_allow_html=True)
        if r["votes"]:
            st.markdown("**How the three networks voted**")
            html = ""
            for net, v in r["votes"].items():
                html += (f'<div class="vote"><span>{net} <span style="color:#6b7773">· weight {v["weight"]:.0%}</span></span>'
                         f'<span>{v["p_tb"]:.0%} · <b class="{v["says"].lower()}">{v["says"]}</b></span></div>'
                         + meter(v["p_tb"], v["threshold"], "#bd4f3b" if v["says"] == "TB" else "#087f78", small=True))
            st.markdown(html, unsafe_allow_html=True)
            agree = len({v["says"] for v in r["votes"].values()}) == 1
            st.caption(("All three networks agree." if agree else "The networks disagree: treat this result with extra caution.")
                       + " Each bar's tick is that network's own threshold.")

    t = r["test"]
    st.markdown(f"""
**What this result means**
- On {results['test_size']:,} unseen test X-rays, {'the ensemble' if r['model_key'] == 'ensemble' else r['model']} caught **{t['Sensitivity'][0]:.0%}** of TB cases and correctly cleared
  **{t['Specificity'][0]:.0%}** of normal ones.
- When it says **TB**, it is right **{t['Precision'][0]:.0%}** of the time; when it says **Normal**, **{t['NPV'][0]:.0%}** of the time.
""" + (f"- It {'also passes' if r['p_tb'] >= r['high_spec_threshold'] else 'does not pass'} the stricter "
       f"high-specificity rule (TB probability ≥ {r['high_spec_threshold']:.0%}), which keeps false alarms to about 5%."
       if is_tb else ""))

    with st.expander("Image and processing details"):
        w, h = r["original_size"]
        st.markdown(f"""
| | |
|---|---|
| Original size | {w} × {h} px ({r['original_mode']}) |
| Model input | 512 × 512 grayscale |
| Detected polarity | {r['polarity']} |
| Model | {r['model']}: {r['blurb']} |
| Calibration | temperature-scaled probability, threshold {r['threshold']:.2f} fixed on validation data |
| Processing time | {r['seconds'] * 1000:.0f} ms on {r['device'].upper()} |
""")

    st.markdown("##### Next steps")
    if is_tb:
        st.markdown("A positive screen should lead to **clinical assessment and confirmatory testing** (sputum smear, culture, "
                    "or a molecular test such as Xpert MTB/RIF) under local TB guidance.")
    else:
        st.markdown("A negative screen **does not rule out TB**. If there are symptoms (cough over 2 weeks, fever, night sweats, "
                    "weight loss) or known exposure, seek clinical assessment.")
    st.caption("PulmoSight is a research and portfolio project, not a medical device. It does not provide a diagnosis.")


PAGES = {
    "intro": st.Page(page_intro, title="Introduction", icon="🫁", default=True),
    "compare": st.Page(page_compare, title="Model comparison", icon="📊", url_path="comparison"),
    "test": st.Page(page_test, title="Test an X-ray", icon="🩻", url_path="test"),
}
st.navigation(list(PAGES.values()), position="top").run()
