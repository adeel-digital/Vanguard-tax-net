import re
import random
import json
import os
from main import (
    SessionLocal, Citizen, Vehicle, UtilityBill, EntityResolutionEngine,
    build_knowledge_graph, compute_graph_metrics, train_gnn,
    calculate_compliance_deviation_scores, clean_cnic
)

def main():
    print("\n" + "="*60)
    print("FBR TAX INTELLIGENCE PLATFORM - INTERACTIVE INGESTION & AUDIT")
    print("="*60)
    
    name = input("1. Enter Person's Name (English or Urdu): ").strip()
    while not name:
        name = input("Name cannot be empty. Enter Name: ").strip()
        
    cnic = input("2. Enter CNIC (e.g., 37405-1234567-1): ").strip()
    while not re.match(r"^\d{5}-\d{7}-\d$", cnic):
        cnic = input("Invalid CNIC format. Please enter as XXXXX-XXXXXXX-X: ").strip()
        
    try:
        income = float(input("3. Enter FBR Reported Annual Income (PKR): "))
    except ValueError:
        income = 0.0
        
    try:
        tax = float(input("4. Enter Annual Tax Paid (PKR): "))
    except ValueError:
        tax = 0.0
        
    try:
        vehicle_cc = int(input("5. Enter Vehicle Engine Capacity in CC (e.g., 1800): "))
    except ValueError:
        vehicle_cc = 1300
        
    try:
        monthly_bill = float(input("6. Enter Monthly Electricity Bill (PKR): "))
    except ValueError:
        monthly_bill = 15000.0
        
    print("\nIngesting test records into SQLite DB with realistic spelling / data anomalies...")
    session = SessionLocal()
    
    try:
        # Check if citizen already exists and delete to avoid duplicate key issues
        existing = session.query(Citizen).filter(Citizen.cnic == cnic).first()
        if existing:
            print(f"[NOTE] Overwriting existing citizen records for CNIC {cnic}")
            session.delete(existing)
            
        # 1. Insert Canonical Citizen Profile
        address = "House No. 12, Street 3, Sector F-7, Islamabad"
        citizen = Citizen(
            cnic=cnic,
            name=name,
            phone="0300-1234567",
            email=f"{name.lower().replace(' ', '')}@gmail.com",
            city="Islamabad",
            address=address,
            declared_income=income,
            tax_paid=tax,
            compliance_score=0.0,
            risk_band="Low Risk"
        )
        session.add(citizen)
        
        # 2. Insert Vehicle with MANGLED owner_cnic (remove dashes) to test ER
        v_cnic = cnic.replace("-", "")
        plate = f"ICT-{random.randint(100, 999)}-Z"
        if vehicle_cc >= 2700:
            make, model, val = "Toyota", f"Fortuner {vehicle_cc}cc", 16000000.0
        elif vehicle_cc >= 1800:
            make, model, val = "Honda", f"Civic {vehicle_cc}cc", 8000000.0
        else:
            make, model, val = "Suzuki", f"Cultus {vehicle_cc}cc", 4000000.0
            
        vehicle = Vehicle(
            plate_number=plate,
            owner_cnic=v_cnic,
            make=make,
            model=model,
            year=2024,
            value=val
        )
        session.add(vehicle)
        
        # 3. Insert Utility Bills (10 months) with MANGLED owner_cnic (alter last digit) to test ER
        # Also slight address spelling variation ("Hose No." and "Strt")
        u_cnic = cnic[:-1] + str((int(cnic[-1]) + 1) % 10)
        cons_num = f"E{random.randint(100000000, 999999999)}"
        mangled_address = "Hose No. 12, Strt 3, Sector F-7, Islamabad"
        
        for month in range(3, 13):
            variation = random.uniform(0.85, 1.15)
            bill_amt = monthly_bill * variation
            bill = UtilityBill(
                utility_type="Electricity",
                consumer_number=cons_num,
                owner_cnic=u_cnic,
                address=mangled_address,
                amount=bill_amt,
                paid=True,
                billing_month=f"2025-{month:02d}"
            )
            session.add(bill)
            
        session.commit()
        print("[SUCCESS] Test records successfully inserted into SQLite database.")
        
        print("\n" + "="*50)
        print("TRIGGERING FBR INTELLIGENCE PIPELINE FOR ANALYSIS...")
        print("="*50)
        
        # Run ER Engine
        er_engine = EntityResolutionEngine(session)
        
        # Rebuild Graph
        G = build_knowledge_graph(session, er_engine)
        compute_graph_metrics(session, G)
        
        # Retrain / Infer GNN
        train_gnn(session, G)
        
        # Calculate deviation scores
        calculate_compliance_deviation_scores(session, er_engine)
        
        # Commit all scores & updates
        session.commit()
        
        # Refresh the session and query our updated citizen
        cit = session.query(Citizen).filter(Citizen.cnic == cnic).first()
        
        # Output SCENE REPORT
        print("\n" + "="*60)
        print(f"TAX COMPLIANCE SCENE REPORT: {cit.name}")
        print("="*60)
        print(f"CNIC:                   {cit.cnic}")
        print(f"Compliance Risk Score:  {cit.compliance_score:.2f} / 100")
        print(f"Assigned Risk Band:     {cit.risk_band}")
        print(f"GNN Anomaly Probability: {cit.anomaly_probability:.4f}")
        print(f"PageRank Centrality:    {cit.graph_pagerank:.6f}")
        print(f"Degree Centrality:      {cit.graph_degree:.6f}")
        
        # Connected Nodes in Knowledge Graph
        print("\nConnected Nodes in Knowledge Graph (Resolved Entities):")
        cit_node = f"Citizen_{cit.cnic}"
        if G.has_node(cit_node):
            neighbors = list(G.successors(cit_node)) + list(G.predecessors(cit_node))
            neighbors = list(set(neighbors))
            if neighbors:
                for n in neighbors:
                    n_label = G.nodes[n].get("label", "Asset")
                    if n_label == "Vehicle":
                        details = f"Vehicle: {G.nodes[n].get('make')} {G.nodes[n].get('model')} (Value: PKR {G.nodes[n].get('value'):,.0f})"
                    elif n_label == "Property":
                        details = f"Property: {G.nodes[n].get('address')} (Value: PKR {G.nodes[n].get('value'):,.0f})"
                    elif n_label == "UtilityBill":
                        details = f"Utility Meter: {n.replace('Utility_', '')} (Annual Bill: PKR {G.nodes[n].get('annual_bill'):,.0f})"
                    elif n_label == "BankAccount":
                        details = f"Bank Account: {G.nodes[n].get('bank_name')} (Balance: PKR {G.nodes[n].get('average_balance'):,.0f})"
                    elif n_label == "TravelRecord":
                        details = f"Travel to: {G.nodes[n].get('destination')} (Cost: PKR {G.nodes[n].get('cost'):,.0f})"
                    else:
                        details = f"Other Node ({n})"
                    print(f"  - [{n_label}] -> {details}")
            else:
                print("  - No connected asset nodes found.")
        else:
            print("  - Citizen node not found in Knowledge Graph.")
            
        # Audit Trail (Explainable AI)
        print("\nStep-by-step Compliance Audit Trail:")
        if cit.xai_explanations:
            try:
                xai_data = json.loads(cit.xai_explanations)
                for r in xai_data.get("reasons", []):
                    print(f"  [+] {r}")
                print(f"\nConclusion: {xai_data.get('conclusion')}")
            except Exception:
                print("  - Could not parse XAI explanations.")
        else:
            print("  - No XAI explanations found.")
            
        print("="*60)
        
    except Exception as e:
        session.rollback()
        print(f"\n[ERROR] Pipeline Ingestion / Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    main()
