from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric
import pandas as pd
import os

def run_drift_report(reference_path, current_data, output_html="drift_report.html"):
    """
    Compare les données actuelles aux données de référence.
    Déclenche une alerte si dérive > 30% des features.
    """
    if not os.path.exists(reference_path):
        print(f"❌ Erreur : Le fichier de référence {reference_path} est introuvable.")
        return False, 0.0

    reference = pd.read_csv(reference_path)

    # On définit le rapport
    report = Report(metrics=[
        DataDriftPreset(drift_share=0.3),
        DataQualityPreset(),
        DatasetDriftMetric()
    ])

    report.run(reference_data=reference, current_data=current_data)
    
    # Export HTML pour les rapports d'artefacts GitHub/DagsHub
    report.save_html(output_html)

    # Extraction sécurisée des résultats
    result = report.as_dict()
    
    drift_detected = False
    drift_share = 0.0

    # On cherche dynamiquement la métrique pour éviter le KeyError par index
    for metric in result["metrics"]:
        if metric["metric"] == "DatasetDriftMetric":
            drift_detected = metric["result"]["dataset_drift"]
            drift_share = metric["result"]["share_of_drifted_columns"]
            break

    print(f"📊 Drift detected: {drift_detected}")
    print(f"📉 Drifted features: {drift_share*100:.1f}%")

    if drift_detected:
        print("⚠ ALERTE: Dérive statistique détectée sur le dataset.")

    return drift_detected, drift_share

if __name__ == "__main__":
    # Bloc de test local
    import numpy as np
    # On s'assure que le dossier artifacts existe pour le test local
    path = "artifacts/reference_data.csv"
    if os.path.exists(path):
        ref = pd.read_csv(path)
        current = ref.copy()
        # Simulation de drift massif
        current.iloc[:, :10] += np.random.normal(5, 2, (len(current), 10))
        run_drift_report(path, current)
