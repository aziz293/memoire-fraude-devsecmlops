from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric
import pandas as pd, json

def run_drift_report(reference_path, current_data, output_html="drift_report.html"):
    """
    Compare les données actuelles aux données de référence.
    Déclenche une alerte si dérive > 30% des features.
    """
    reference = pd.read_csv(reference_path)

    report = Report(metrics=[
        DataDriftPreset(drift_share=0.3),
        DataQualityPreset(),
        DatasetDriftMetric()
    ])

    report.run(reference_data=reference, current_data=current_data)
    report.save_html(output_html)

    result = report.as_dict()
    drift_detected = result["metrics"][2]["result"]["dataset_drift"]
    drift_share = result["metrics"][2]["result"]["share_of_drifted_columns"]

    print(f"Drift detected: {drift_detected}")
    print(f"Drifted features: {drift_share*100:.1f}%")

    if drift_detected:
        print("⚠ ALERTE: Dérive détectée — réentraînement recommandé")

    return drift_detected, drift_share

if __name__ == "__main__":
    # Simuler une dérive en ajoutant du bruit aux données
    import numpy as np
    ref = pd.read_csv("artifacts/reference_data.csv")
    current = ref.copy()
    current += np.random.normal(0, 2, current.shape)  # Dérive artificielle
    run_drift_report("artifacts/reference_data.csv", current)

