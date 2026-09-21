
import gradio as gr
import joblib, pandas as pd, numpy as np

ART = joblib.load('breast_cancer_model.pkl')
MODEL, THRESH, FEATURES = ART['model'], ART['threshold'], ART['feature_names']
QUADRANTS = ['Lower Inner', 'Lower Outer', 'Unknown', 'Upper Inner', 'Upper Outer']
MODEL_NAME = 'Decision Tree'
CV_MEAN, CV_STD = 0.840, 0.036

def predict_single(age, menopause, tumor_size, inv_nodes, breast, metastasis, quadrant, history):
    rec = pd.DataFrame([{
        'Age': age, 'Menopause': int(menopause), 'Tumor Size (cm)': float(tumor_size),
        'Inv-Nodes': int(inv_nodes), 'Breast': breast, 'Metastasis': int(metastasis),
        'Breast Quadrant': quadrant, 'History': int(history)
    }])[FEATURES]

    p = float(MODEL.predict_proba(rec)[:, 1][0])
    malignant = p >= THRESH

    label, color = ('Malignant', '#c0392b') if malignant else ('Benign', '#27ae60')
    certainty = 'High' if abs(p - 0.5) > 0.3 else 'Moderate'
    bar_pct = int(p * 100)

    html = f"""
    <div style="font-family:system-ui;padding:18px;border-radius:10px;
                border:2px solid {color};background:#fafafa">
      <div style="font-size:26px;font-weight:700;color:{color}">{label}</div>
      <div style="margin:14px 0 6px;font-size:13px;color:#555">
        Probability of malignancy: <b>{p:.1%}</b> &nbsp;|&nbsp; Certainty: {certainty}
      </div>
      <div style="height:22px;background:#e8e8e8;border-radius:11px;overflow:hidden">
        <div style="height:100%;width:{bar_pct}%;background:{color}"></div>
      </div>
      <div style="margin-top:6px;font-size:11px;color:#777">
        Decision threshold: {THRESH:.0%} — set below 50% to prioritise sensitivity
      </div>
      <div style="margin-top:14px;padding:10px;background:#fff8e1;border-radius:6px;
                  font-size:12px;color:#6b5500">
        Educational coursework demo. Not a medical device and not a substitute for
        clinical diagnosis.
      </div>
    </div>"""
    return html


def predict_batch(file):
    if file is None:
        return None, 'Upload a CSV first.'
    df = pd.read_csv(file.name)
    df.columns = df.columns.str.strip()

    missing = set(FEATURES) - set(df.columns)
    if missing:
        return None, f'Missing columns: {sorted(missing)}'

    proba = MODEL.predict_proba(df[FEATURES])[:, 1]
    out = df.copy()
    out['P(Malignant)'] = proba.round(3)
    out['Prediction'] = np.where(proba >= THRESH, 'Malignant', 'Benign')

    out.to_csv('predictions.csv', index=False)
    n_mal = (out['Prediction'] == 'Malignant').sum()
    return out.head(50), f'{len(out)} rows scored — {n_mal} flagged malignant. Saved to predictions.csv'


