import os
import time
from flask import Flask, send_from_directory, jsonify, request
from federated_aggregator import FederatedAggregator, train_all_nodes

app = Flask(__name__, static_folder="../frontend")

# Initialize global aggregator state
print("Initializing Federated Network...")
# Ensure we are in the backend directory so paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))
apis = train_all_nodes(".")
aggregator = FederatedAggregator(
    reference_registry_path="data/hashed_id_registry.csv",
    aggregation_method="weighted_average"
)
for inst_type, api in apis.items():
    aggregator.register_node(inst_type, api)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/status")
def status():
    return jsonify({
        "status": "online",
        "nodes": list(aggregator.nodes.keys()),
        "trust_scores": aggregator.node_trust_scores
    })

@app.route("/api/run-inference", methods=["POST"])
def run_inference():
    import pandas as pd
    # Select 10 random customers to score for speed
    registry = pd.read_csv("data/hashed_id_registry.csv")
    sample_ids = registry["hashed_customer_id"].sample(10).tolist()
    
    results = aggregator.run_inference_round(sample_ids, round_id=f"web_round_{time.time()}")
    
    # Format output
    output = []
    for hid, res in results.items():
        output.append({
            "customer_id": hid[:12] + "...",
            "final_score": res.final_score,
            "risk_tier": res.risk_tier,
            "alerts": [a.alert_type for a in res.alerts]
        })
    
    return jsonify({
        "success": True,
        "results": output,
        "trust_scores": aggregator.node_trust_scores,
        "blocked_nodes": list(aggregator.blocked_nodes)
    })

@app.route("/api/simulate-attack", methods=["POST"])
def simulate_attack_endpoint():
    from local_nodes import RogueNodeAPI, DifferentialPrivacyWrapper
    import pandas as pd
    
    # Temporarily replace insurer with rogue node
    target = "insurer"
    legit_api = aggregator.nodes.get(target)
    
    dp_wrapper = DifferentialPrivacyWrapper(noise_std=0.05, round_precision=0.1)
    rogue_api = RogueNodeAPI(legit_api.trainer, dp_wrapper, attack_type="inversion")
    
    aggregator.nodes[target] = rogue_api
    
    # Reset trust
    aggregator.node_trust_scores[target] = 1.0
    if target in aggregator.blocked_nodes:
        aggregator.blocked_nodes.remove(target)
        
    registry = pd.read_csv("data/hashed_id_registry.csv")
    sample_ids = registry["hashed_customer_id"].sample(15).tolist()
    
    results = aggregator.run_inference_round(sample_ids, round_id=f"attack_round_{time.time()}")
    
    # Restore legit node
    aggregator.nodes[target] = legit_api
    
    output = []
    for hid, res in results.items():
        output.append({
            "customer_id": hid[:12] + "...",
            "final_score": res.final_score,
            "risk_tier": res.risk_tier,
            "alerts": [a.alert_type for a in res.alerts]
        })
        
    return jsonify({
        "success": True,
        "message": "Simulated Score Inversion Attack on Insurer Node",
        "results": output,
        "trust_scores": aggregator.node_trust_scores,
        "blocked_nodes": list(aggregator.blocked_nodes)
    })

if __name__ == "__main__":
    print("=========================================================")
    print("  PS2 — Federated Ensemble Scoring")
    print("  Starting Flask Web Server...")
    print("  Open http://localhost:5000 in your browser to view the frontend")
    print("=========================================================")
    app.run(host="127.0.0.1", port=5000, debug=False)

