import os
import json
import datetime
from code.ipso_ga import run_ipso_ga


CITY = os.environ.get("CITY", "Tegal")


def generate_run_id(city: str, metrics: dict) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    cost = int(metrics.get("cost", 0))
    return f"{city}_{ts}_{cost}"


def main():
    print("PenjuruBus IPSO-GA started...")

    result = run_ipso_ga(output_dir=f"output_{CITY}")
    print("IPSO-GA complete.")

    run_id = generate_run_id(CITY, result["metrics"])

    # Ekspor ringkas (bisa di-baca frontend)
    summary = {
        "city": CITY,
        "run_id": run_id,
        "model": "IPSO-GA",
        "halte_count": len(result["halte_ideal"]),
        "metrics": result["metrics"],
        "halte_geojson": result.get("halte_file", None),
    }
    with open(f"summary_{CITY}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Summary saved.")


if __name__ == "__main__":
    main()