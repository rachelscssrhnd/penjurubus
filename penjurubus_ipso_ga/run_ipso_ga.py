import os
import json
import datetime
from code.ipso_ga import run_ipso_ga
from code.azure_aml_logger import log_ipsoga_run


EXPERIMENT = os.environ.get("IPSOGA_EXPERIMENT", "penjurubus-ipsoga-free-2026")
CITY = os.environ.get("CITY", "Tegal")


def generate_run_id(city: str, metrics: dict) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    cost = int(metrics.get("cost", 0))
    return f"{city}_{ts}_{cost}"


def main():
    print("PenjuruBus IPSO‑GA (Azure Free Tier) started...")

    result = run_ipso_ga(output_dir=f"output_{CITY}")
    print("IPSO‑GA complete.")

    run_id = generate_run_id(CITY, result["metrics"])
    try:
        log_ipsoga_run(
            experiment_name=EXPERIMENT,
            run_name=run_id,
            metrics=result["metrics"],
            best_halte=result["halte_ideal"]
        )
    except Exception as e:
        print(f"[Azure ML] Warning: logging error (Free Tier only): {e}")

    # 4. Ekspor ringkas (bisa di‑baca frontend)
    summary = {
        "city": CITY,
        "run_id": run_id,
        "model": "IPSO‑GA (Free Tier)",
        "halte_count": len(result["halte_ideal"]),
        "metrics": result["metrics"],
        "halte_geojson": result.get("halte_file", None),
    }
    with open(f"summary_{CITY}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Summary saved (ready for Azure Functions response / frontend).")


if __name__ == "__main__":
    main()