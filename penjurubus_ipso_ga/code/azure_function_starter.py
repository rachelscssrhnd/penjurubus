import azure.functions as func
import logging
import os
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "penjurubus_ipso_ga"))

from penjurubus_ipso_ga.code.ipso_ga import run_ipso_ga
from penjurubus_ipso_ga.code.azure_aml_logger import log_ipsoga_run
from penjurubus_ipso_ga.run_ipso_ga import EXPERIMENT, CITY, generate_run_id


app = func.FunctionApp()


@app.function_name("GenerateNetwork")
@app.route(route="generate_network", auth_level=func.AuthLevel.FUNCTION, methods=["POST"])
def generate_network(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function (Consumption Plan — FREE TIER eligible) untuk:
    - menerima trigger dari UI (React) saat user klik "Generate Network"
    - menjalankan pipeline: IPSO‑GA + log AML + return status async
    """
    logging.info("Azure Function (Free Tier): GenerateNetwork triggered.")

    body = req.get_json()
    city = body.get("city", CITY)
    experiment = body.get("experiment", EXPERIMENT)

    os.environ["CITY"] = city
    os.environ["IPSOGA_EXPERIMENT"] = experiment

    result = run_ipso_ga(output_dir=f"output_{city}")

    run_id = generate_run_id(city, result["metrics"])
    try:
        log_ipsoga_run(
            experiment_name=experiment,
            run_name=run_id,
            metrics=result["metrics"],
            best_halte=result["halte_ideal"]
        )
    except Exception as e:
        logging.warning(f"AML logging (Free Tier): {e}")

    status = {
        "job_id": run_id,
        "status": "completed",
        "city": city,
        "model": "IPSO‑GA (Free)",
        "halte_count": len(result["halte_ideal"]),
        "coverage": result["metrics"].get("coverage", 0.0),
        "route_km": result["metrics"].get("route_km", 0.0),
        "halte_geojson_url": result.get("halte_file", ""),
    }

    logging.info(f"Function completed: {status}")

    return func.HttpResponse(
        json.dumps(status, ensure_ascii=False),
        mimetype="application/json",
        status_code=202,
    )