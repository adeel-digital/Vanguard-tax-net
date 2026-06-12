import re
import json
import random
from main import (
    SessionLocal, Citizen, Vehicle, UtilityBill, EntityResolutionEngine,
    build_knowledge_graph, compute_graph_metrics, train_gnn,
    calculate_compliance_deviation_scores, clean_cnic
)

def run_govtech_demo():
    print("=" * 70)
    print("FBR GOVTECH RESEARCH DEPLOYMENT - INGESTING CUSTOM DEMO CASE")
    print("=" * 70)
    
    session = SessionLocal()
    
    # Target Demo Case Details
    canonical_cnic = "42101-1234567-1"
    canonical_name = "Adeel Haider"
    reported_income = 500000.0  # Low income: 500k PKR/year
    tax_paid = 15000.0
    
    try:
        # 1. Clean existing records for this demo identity to prevent duplicate key errors
        print(f"\n[1/4] Cleaning existing demo records for CNIC {canonical_cnic}...")
        existing_citizen = session.query(Citizen).filter(Citizen.cnic == canonical_cnic).first()
        if existing_citizen:
            session.delete(existing_citizen)
            
        mangled_v_cnic = canonical_cnic.replace("-", "") # 4210112345671
        existing_vehicles = session.query(Vehicle).filter(Vehicle.owner_cnic == mangled_v_cnic).all()
        for v in existing_vehicles:
            session.delete(v)
            
        mangled_u_cnic = canonical_cnic[:-1] + "2" # 42101-1234567-2
        existing_bills = session.query(UtilityBill).filter(UtilityBill.owner_cnic == mangled_u_cnic).all()
        for b in existing_bills:
            session.delete(b)
            
        session.commit()
        
        # 2. Ingest Canonical Citizen Profile
        print(f"[2/4] Ingesting canonical Citizen profile: {canonical_name}...")
        citizen = Citizen(
            cnic=canonical_cnic,
            name=canonical_name,
            phone="0321-5558899",
            email="adeel.haider@gov.pk",
            city="Islamabad",
            address="House No. 45, Street 12, Sector G-11, Islamabad",
            declared_income=reported_income,
            tax_paid=tax_paid,
            compliance_score=0.0,
            risk_band="Low Risk"
        )
        session.add(citizen)
        
        # 3. Ingest Mangled Vehicle Registry (Excise DB)
        # Registered under mangled CNIC (no dashes) and name variation 'Adil Haidar'
        # Model specifies CC and high value (8,000,000 PKR)
        print(f"[3/4] Ingesting mangled Excise Registry vehicle: plate 'ICT-505-A' owned by raw CNIC '{mangled_v_cnic}' (Adil Haidar)...")
        vehicle = Vehicle(
            plate_number="ICT-505-A",
            owner_cnic=mangled_v_cnic,
            make="Honda",
            model="Civic 2000cc",
            year=2024,
            value=8000000.0
        )
        session.add(vehicle)
        
        # 4. Ingest Mangled Utility Bills (LESC / K-Electric Registry)
        # Registered under mangled CNIC (altered last digit) and name variation 'Adeel H.'
        # Injects 10 months of electricity bills totaling 300,000 PKR
        print(f"[4/4] Ingesting mangled Utility Meter: consumer 'E9028194' owned by raw CNIC '{mangled_u_cnic}' (Adeel H.)...")
        cons_number = "E9028194"
        mangled_address = "Hose No. 45, Strt 12, Sector G-11, Islamabad" # typos
        
        for month in range(3, 13):
            # Ingest bills of approx 30k monthly
            variation = random.uniform(0.90, 1.10)
            bill_amt = 30000.0 * variation
            bill = UtilityBill(
                utility_type="Electricity",
                consumer_number=cons_number,
                owner_cnic=mangled_u_cnic,
                address=mangled_address,
                amount=bill_amt,
                paid=True,
                billing_month=f"2025-{month:02d}"
            )
            session.add(bill)
            
        session.commit()
        print("\n[SUCCESS] Custom demo records inserted successfully into database.")
        
        # Triggering Pipeline
        print("\n" + "="*50)
        print("TRIGGERING PIPELINE: RUNNING ENTITY RESOLUTION & AI SCORING...")
        print("="*50)
        
        er_engine = EntityResolutionEngine(session)
        
        # Verify that ER successfully fuzzy-matches 'Adil Haidar' and 'Adeel H.' back to 'Adeel Haider'
        print("\nFuzzy Matching Resolver Test:")
        resolved_v, score_v = er_engine.resolve_citizen(mangled_v_cnic, query_name="Adil Haidar")
        if resolved_v:
            print(f"  - Resolved vehicle owner CNIC '{mangled_v_cnic}' ('Adil Haidar') -> '{resolved_v.name}' ({resolved_v.cnic}) with confidence {score_v}")
            
        resolved_u, score_u = er_engine.resolve_citizen(mangled_u_cnic, query_name="Adeel H.", query_address=mangled_address)
        if resolved_u:
            print(f"  - Resolved utility meter owner CNIC '{mangled_u_cnic}' ('Adeel H.') -> '{resolved_u.name}' ({resolved_u.cnic}) with confidence {score_u}")
            
        # Rebuild Graph
        G = build_knowledge_graph(session, er_engine)
        compute_graph_metrics(session, G)
        
        # Train / Infer GNN
        train_gnn(session, G)
        
        # Calculate deviation scores
        calculate_compliance_deviation_scores(session, er_engine)
        
        session.commit()
        
        # Query updated citizen details
        cit = session.query(Citizen).filter(Citizen.cnic == canonical_cnic).first()
        
        # Print Scene Report
        print("\n" + "="*60)
        print(f"DEMO COMPLIANCE SCENE REPORT: {cit.name}")
        print("="*60)
        print(f"CNIC:                   {cit.cnic}")
        print(f"Compliance Risk Score:  {cit.compliance_score:.2f} / 100")
        print(f"Assigned Risk Band:     {cit.risk_band}")
        print(f"GNN Anomaly Probability: {cit.anomaly_probability:.4f}")
        print(f"PageRank Centrality:    {cit.graph_pagerank:.6f}")
        print(f"Degree Centrality:      {cit.graph_degree:.6f}")
        
        print("\nResolved Connected Graph Nodes:")
        cit_node = f"Citizen_{cit.cnic}"
        if G.has_node(cit_node):
            neighbors = list(G.successors(cit_node)) + list(G.predecessors(cit_node))
            neighbors = list(set(neighbors))
            for n in neighbors:
                n_label = G.nodes[n].get("label", "Asset")
                if n_label == "Vehicle":
                    details = f"Vehicle: {G.nodes[n].get('make')} {G.nodes[n].get('model')} (Value: PKR {G.nodes[n].get('value'):,.0f})"
                elif n_label == "UtilityBill":
                    details = f"Utility Meter: {n.replace('Utility_', '')} (Annual Bill: PKR {G.nodes[n].get('annual_bill'):,.0f})"
                else:
                    details = f"Other Asset Node ({n})"
                print(f"  - [{n_label}] -> {details}")
                
        print("\nCompliance Audit Trail:")
        if cit.xai_explanations:
            xai_data = json.loads(cit.xai_explanations)
            for r in xai_data.get("reasons", []):
                print(f"  [+] {r}")
            print(f"\nConclusion: {xai_data.get('conclusion')}")
            
        print("="*60)
        print("\nDemonstration ingestion complete. You can now refresh and search this taxpayer in the UI!")
        
    except Exception as e:
        session.rollback()
        print(f"\n[ERROR] Demo workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    run_govtech_demo()
