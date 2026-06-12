import os
import re
import json
import sqlite3
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import torch
import torch.nn as nn
import torch.nn.functional as F
import Levenshtein

# -------------------------------------------------------------
# 1. Database Connections and SQLAlchemy Schema
# -------------------------------------------------------------
DATABASE_PATH = "tax_graph_ai.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

Base = declarative_base()

class Citizen(Base):
    __tablename__ = "citizens"
    id = Column(Integer, primary_key=True)
    cnic = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    city = Column(String, nullable=True)
    address = Column(String, nullable=True)
    declared_income = Column(Float, default=0.0)
    tax_paid = Column(Float, default=0.0)
    compliance_score = Column(Float, default=0.0)
    risk_band = Column(String, default="Low Risk")
    anomaly_probability = Column(Float, default=0.0)
    xai_explanations = Column(Text, nullable=True)
    graph_pagerank = Column(Float, default=0.0)
    graph_degree = Column(Float, default=0.0)
    graph_community = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=True)

class Vehicle(Base):
    __tablename__ = "vehicles"
    id = Column(Integer, primary_key=True)
    plate_number = Column(String, nullable=False)
    owner_cnic = Column(String, nullable=False)
    make = Column(String, nullable=False)
    model = Column(String, nullable=False)
    year = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)

class Property(Base):
    __tablename__ = "properties"
    id = Column(Integer, primary_key=True)
    owner_cnic = Column(String, nullable=False)
    address = Column(String, nullable=False)
    city = Column(String, nullable=False)
    property_type = Column(String, nullable=False)
    area = Column(String, nullable=False)
    value = Column(Float, nullable=False)

class UtilityBill(Base):
    __tablename__ = "utility_bills"
    id = Column(Integer, primary_key=True)
    utility_type = Column(String, nullable=False)
    consumer_number = Column(String, nullable=False)
    owner_cnic = Column(String, nullable=False)
    address = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    paid = Column(Boolean, default=True)
    billing_month = Column(String, nullable=False)

class BankAccount(Base):
    __tablename__ = "bank_accounts"
    id = Column(Integer, primary_key=True)
    account_number = Column(String, unique=True, nullable=False)
    owner_cnic = Column(String, nullable=False)
    bank_name = Column(String, nullable=False)
    average_balance = Column(Float, default=0.0)
    transaction_volume = Column(Float, default=0.0)
    cash_withdrawals = Column(Float, default=0.0)
    international_transfers = Column(Float, default=0.0)

class TravelRecord(Base):
    __tablename__ = "travel_records"
    id = Column(Integer, primary_key=True)
    passport_number = Column(String, nullable=False)
    owner_cnic = Column(String, nullable=False)
    travel_date = Column(String, nullable=False)
    destination = Column(String, nullable=False)
    trip_type = Column(String, default="Leisure")
    cost = Column(Float, default=0.0)

class EntityResolutionTruth(Base):
    __tablename__ = "entity_resolution_truth"
    id = Column(Integer, primary_key=True)
    record_type = Column(String, nullable=False)
    record_key = Column(String, nullable=False)
    raw_cnic = Column(String, nullable=False)
    raw_name = Column(String, nullable=True)
    raw_address = Column(String, nullable=True)
    correct_cnic = Column(String, nullable=False)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# -------------------------------------------------------------
# 2. Optimized Entity Resolution Logic (with Candidate Blocking)
# -------------------------------------------------------------
URDU_TO_ENG_DICT = {
    "محمد": "Muhammad",
    "احمد": "Ahmad",
    "علی": "Ali",
    "فاطمہ": "Fatima",
    "عمران": "Imran",
    "خان": "Khan",
    "آصف": "Asif",
    "محمود": "Mahmood",
    "شاہ": "Shah",
    "حمزہ": "Hamza",
    "حسین": "Hussain",
    "منیب": "Muneeb",
    "ملک": "Malik",
    "بی بی": "Bibi",
    "بلال": "Bilal",
    "قریشی": "Qureshi",
    "طارق": "Tariq",
    "جمیل": "Jameel",
    "زینب": "Zainab"
}

def clean_cnic(cnic):
    if not cnic:
        return ""
    return re.sub(r"\D", "", str(cnic))

def clean_name(name):
    if not name:
        return ""
    name = str(name).strip()
    if any(u"\u0600" <= c <= u"\u06FF" for c in name):
        words = name.split()
        translated = [URDU_TO_ENG_DICT.get(w, w) for w in words]
        name = " ".join(translated)
        
    name = name.lower()
    name = re.sub(r"\b(m\.)\b", "muhammad", name)
    name = re.sub(r"\b(md\.)\b", "muhammad", name)
    name = re.sub(r"\b(ch\.)\b", "chaudhry", name)
    name = re.sub(r"\b(syed)\b", "", name)
    name = re.sub(r"\b(ch)\b", "chaudhry", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def levenshtein_similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def token_similarity(s1, s2):
    return fuzz.token_sort_ratio(s1, s2) / 100.0

def tfidf_address_similarity(addr1, addr2, vectorizer=None):
    if not addr1 or not addr2:
        return 0.0
    try:
        if vectorizer is None:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            tfidf = vectorizer.fit_transform([addr1.lower(), addr2.lower()])
            return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        else:
            tfidf = vectorizer.transform([addr1.lower(), addr2.lower()])
            return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return levenshtein_similarity(addr1, addr2)

class EntityResolutionEngine:
    def __init__(self, session):
        self.session = session
        print("Caching Citizen registry profiles for fast lookup...")
        self.citizens = session.query(Citizen).all()
        self.cnic_map = {clean_cnic(c.cnic): c for c in self.citizens}
        
    def resolve_citizen(self, query_cnic, query_name="", query_address="", query_phone=""):
        cleaned_query_cnic = clean_cnic(query_cnic)
        cleaned_query_name = clean_name(query_name)
        
        # 1. Direct Cleaned CNIC Match (O(1) lookup)
        if cleaned_query_cnic in self.cnic_map:
            citizen = self.cnic_map[cleaned_query_cnic]
            cleaned_citizen_name = clean_name(citizen.name)
            name_score = token_similarity(cleaned_query_name, cleaned_citizen_name) if query_name else 1.0
            if name_score > 0.6:
                return citizen, round(0.95 + (0.05 * name_score), 3)
            else:
                return citizen, round(0.70 + (0.20 * name_score), 3)
                
        # 2. Inexact Match with Candidate Blocking (O(K) instead of O(N))
        candidates = []
        if cleaned_query_cnic:
            for cnic_key, citizen in self.cnic_map.items():
                if Levenshtein.distance(cleaned_query_cnic, cnic_key) <= 2:
                    candidates.append((cnic_key, citizen))
        
        # Fallback blocking on phone/name if no CNIC match
        if not candidates:
            for cnic_key, citizen in self.cnic_map.items():
                # Phone match
                if query_phone and citizen.phone:
                    clean_q_p = re.sub(r"\D", "", query_phone)
                    clean_c_p = re.sub(r"\D", "", citizen.phone)
                    if clean_q_p[-7:] == clean_c_p[-7:]:
                        candidates.append((cnic_key, citizen))
                        continue
                # Name overlap match
                if query_name and citizen.name:
                    q_words = set(cleaned_query_name.split())
                    c_words = set(clean_name(citizen.name).split())
                    stop_words = {"muhammad", "ahmad", "ali", "khan", "syed", "chaudhry"}
                    if (q_words - stop_words) & (c_words - stop_words):
                        candidates.append((cnic_key, citizen))
                        
        if not candidates:
            return None, 0.0
            
        best_candidate = None
        best_score = 0.0
        
        for cnic_key, citizen in candidates:
            cnic_score = 0.0
            name_score = 0.0
            addr_score = 0.0
            phone_score = 0.0
            
            if cleaned_query_cnic:
                cnic_score = levenshtein_similarity(cleaned_query_cnic, cnic_key)
                
            if query_phone and citizen.phone:
                clean_q_p = re.sub(r"\D", "", query_phone)
                clean_c_p = re.sub(r"\D", "", citizen.phone)
                if clean_q_p == clean_c_p:
                    phone_score = 1.0
                elif clean_q_p[-7:] == clean_c_p[-7:]:
                    phone_score = 0.85
                    
            if query_name:
                cleaned_citizen_name = clean_name(citizen.name)
                name_score = token_similarity(cleaned_query_name, cleaned_citizen_name)
                
            if query_address and citizen.address:
                addr_score = tfidf_address_similarity(query_address, citizen.address)
                
            weights = {"cnic": 0.50, "name": 0.25, "address": 0.15, "phone": 0.10}
            current_weights = {}
            if cleaned_query_cnic: current_weights["cnic"] = weights["cnic"]
            if query_name: current_weights["name"] = weights["name"]
            if query_address and citizen.address: current_weights["address"] = weights["address"]
            if query_phone and citizen.phone: current_weights["phone"] = weights["phone"]
            
            total_w = sum(current_weights.values())
            if total_w == 0:
                continue
                
            norm_w = {k: v / total_w for k, v in current_weights.items()}
            
            composite_score = (
                norm_w.get("cnic", 0.0) * cnic_score +
                norm_w.get("name", 0.0) * name_score +
                norm_w.get("address", 0.0) * addr_score +
                norm_w.get("phone", 0.0) * phone_score
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_candidate = citizen
                
        if best_score > 0.65 and best_candidate:
            return best_candidate, round(best_score, 3)
            
        return None, 0.0

def run_entity_resolution_evaluation(session, er_engine):
    print("\n--- Running Entity Resolution Ground-Truth Evaluation ---")
    truths = session.query(EntityResolutionTruth).all()
    if not truths:
        print("[WARNING] No ground-truth records found in entity_resolution_truth.")
        return
        
    tp, fp, fn = 0, 0, 0
    for t in truths:
        resolved, score = er_engine.resolve_citizen(
            query_cnic=t.raw_cnic,
            query_name=t.raw_name or "",
            query_address=t.raw_address or ""
        )
        if resolved:
            if resolved.cnic == t.correct_cnic:
                tp += 1
            else:
                fp += 1
        else:
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"Total Ground-Truth Checked: {len(truths)}")
    print(f"True Positives (TP):      {tp}")
    print(f"False Positives (FP):     {fp}")
    print(f"False Negatives (FN):     {fn}")
    print(f"ER Precision:             {precision:.4f}")
    print(f"ER Recall:                {recall:.4f}")
    print(f"ER F1-Score:              {f1:.4f}")
    print("------------------------------------------------------\n")

# -------------------------------------------------------------
# 3. Knowledge Graph Construction
# -------------------------------------------------------------
def build_knowledge_graph(session, er_engine):
    print("Building NetworkX Knowledge Graph...")
    G = nx.MultiDiGraph()
    
    # 1. Add Citizen nodes
    citizens = session.query(Citizen).all()
    cit_dict = {}
    for c in citizens:
        cit_id = f"Citizen_{c.cnic}"
        props = {
            "label": "Citizen",
            "name": c.name,
            "cnic": c.cnic,
            "city": c.city or "",
            "declared_income": c.declared_income,
            "tax_paid": c.tax_paid,
            "graph_pagerank": c.graph_pagerank,
            "graph_degree": c.graph_degree
        }
        G.add_node(cit_id, **props)
        cit_dict[c.cnic] = cit_id
        
    # Helper to resolve owner CNIC
    def get_resolved_cnic(raw_cnic, address=""):
        resolved, score = er_engine.resolve_citizen(raw_cnic, query_address=address)
        if resolved:
            return resolved.cnic
        cleaned = clean_cnic(raw_cnic)
        if cleaned in er_engine.cnic_map:
            return er_engine.cnic_map[cleaned].cnic
        return None

    # 2. Vehicles
    vehicles = session.query(Vehicle).all()
    for v in vehicles:
        owner_cnic = get_resolved_cnic(v.owner_cnic)
        if owner_cnic:
            v_id = f"Vehicle_{v.plate_number}"
            G.add_node(v_id, label="Vehicle", make=v.make, model=v.model, value=v.value)
            G.add_edge(f"Citizen_{owner_cnic}", v_id, type="OWNS")
            
    # 3. Properties
    properties = session.query(Property).all()
    for p in properties:
        owner_cnic = get_resolved_cnic(p.owner_cnic, address=p.address)
        if owner_cnic:
            p_id = f"Property_{p.id}"
            G.add_node(p_id, label="Property", address=p.address, city=p.city, value=p.value)
            G.add_edge(f"Citizen_{owner_cnic}", p_id, type="OWNS")
            
    # 4. Utility bills (Grouped by consumer number)
    utility_bills = session.query(UtilityBill).all()
    grouped_meters = {}
    for u in utility_bills:
        if u.consumer_number not in grouped_meters:
            grouped_meters[u.consumer_number] = {
                "owner_cnic": u.owner_cnic,
                "address": u.address,
                "utility_type": u.utility_type,
                "total_amount": 0.0
            }
        grouped_meters[u.consumer_number]["total_amount"] += u.amount
        
    for cons_num, gm in grouped_meters.items():
        owner_cnic = get_resolved_cnic(gm["owner_cnic"], address=gm["address"])
        if owner_cnic:
            u_id = f"Utility_{cons_num}"
            G.add_node(u_id, label="UtilityBill", type=gm["utility_type"], annual_bill=gm["total_amount"])
            G.add_edge(f"Citizen_{owner_cnic}", u_id, type="PAID")
            
    # 5. Bank Accounts
    bank_accounts = session.query(BankAccount).all()
    for b in bank_accounts:
        owner_cnic = get_resolved_cnic(b.owner_cnic)
        if owner_cnic:
            b_id = f"BankAccount_{b.account_number}"
            G.add_node(b_id, label="BankAccount", bank_name=b.bank_name, average_balance=b.average_balance, volume=b.transaction_volume)
            G.add_edge(f"Citizen_{owner_cnic}", b_id, type="HAS_ACCOUNT")
            
    # 6. Travel Records
    travels = session.query(TravelRecord).all()
    for t in travels:
        owner_cnic = get_resolved_cnic(t.owner_cnic)
        if owner_cnic:
            t_id = f"Travel_{t.id}"
            G.add_node(t_id, label="TravelRecord", destination=t.destination, cost=t.cost)
            G.add_edge(f"Citizen_{owner_cnic}", t_id, type="TRAVELLED")
            
    print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def compute_graph_metrics(session, G):
    print("Computing graph centrality metrics...")
    if len(G) == 0:
        return
        
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
    except Exception as e:
        print(f"PageRank error: {e}")
        pagerank = {node: 1.0/len(G) for node in G.nodes}
        
    degree_centrality = nx.degree_centrality(G)
    
    G_undirected = nx.Graph(G)
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G_undirected)
    except Exception:
        communities = list(nx.connected_components(G_undirected))
        
    node_to_comm = {}
    for c_idx, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = c_idx
            
    citizens = session.query(Citizen).all()
    for c in citizens:
        node_id = f"Citizen_{c.cnic}"
        pr = pagerank.get(node_id, 0.0)
        deg = degree_centrality.get(node_id, 0.0)
        comm = node_to_comm.get(node_id, 0)
        
        c.graph_pagerank = float(pr * 1000.0)
        c.graph_degree = float(deg * 100.0)
        c.graph_community = int(comm)
        
    session.commit()
    print("[SUCCESS] Graph metrics saved to Citizen DB.")

# -------------------------------------------------------------
# 4. PyTorch GNN Anomaly Model Implementation
# -------------------------------------------------------------
class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        device = x.device
        
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index_with_loops = torch.cat([edge_index, loop_index], dim=1)
        
        row, col = edge_index_with_loops[0], edge_index_with_loops[1]
        
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        x_transformed = self.linear(x)
        neighbor_features = x_transformed[col]
        scaled_features = neighbor_features * norm.unsqueeze(-1)
        
        out = torch.zeros(num_nodes, x_transformed.size(1), device=device)
        out.scatter_add_(0, row.unsqueeze(-1).repeat(1, x_transformed.size(1)), scaled_features)
        return out

class TaxGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=1):
        super(TaxGCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.classifier = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.classifier(h)
        return torch.sigmoid(h)

def train_gnn(session, G):
    print("Preparing features for PyTorch GCN Anomaly Detector...")
    node_to_idx = {node: i for i, node in enumerate(G.nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    num_nodes = len(G)
    
    x_raw = np.zeros((num_nodes, 8))
    y_raw = np.zeros(num_nodes)
    citizen_mask = np.zeros(num_nodes, dtype=bool)
    
    citizens = session.query(Citizen).all()
    citizens_db = {c.cnic: c for c in citizens}
    
    vehicles = session.query(Vehicle).all()
    properties = session.query(Property).all()
    utility_bills = session.query(UtilityBill).all()
    bank_accounts = session.query(BankAccount).all()
    travels = session.query(TravelRecord).all()
    
    v_map = {}
    for v in vehicles: v_map.setdefault(clean_cnic(v.owner_cnic), []).append(v)
    p_map = {}
    for p in properties: p_map.setdefault(clean_cnic(p.owner_cnic), []).append(p)
    u_map = {}
    for u in utility_bills: u_map.setdefault(clean_cnic(u.owner_cnic), []).append(u)
    b_map = {}
    for b in bank_accounts: b_map.setdefault(clean_cnic(b.owner_cnic), []).append(b)
    t_map = {}
    for t in travels: t_map.setdefault(clean_cnic(t.owner_cnic), []).append(t)
    
    for node, idx in node_to_idx.items():
        node_type = G.nodes[node].get("label", "Asset")
        x_raw[idx, 6] = G.nodes[node].get("graph_pagerank", 0.0)
        x_raw[idx, 7] = G.nodes[node].get("graph_degree", 0.0)
        
        if node_type == "Citizen":
            cnic = G.nodes[node].get("cnic")
            c = citizens_db.get(cnic)
            if c:
                c_cnic = clean_cnic(cnic)
                x_raw[idx, 0] = c.declared_income
                x_raw[idx, 1] = c.tax_paid
                
                total_assets = sum(v.value for v in v_map.get(c_cnic, [])) + sum(p.value for p in p_map.get(c_cnic, []))
                x_raw[idx, 2] = total_assets
                x_raw[idx, 3] = sum(u.amount for u in u_map.get(c_cnic, []))
                x_raw[idx, 4] = sum(t.cost for t in t_map.get(c_cnic, []))
                x_raw[idx, 5] = sum(b.transaction_volume for b in b_map.get(c_cnic, []))
                
                lifestyle = (total_assets * 0.15) + x_raw[idx, 3] + x_raw[idx, 4]
                if (c.declared_income < 1000000 and lifestyle > 3000000) or (lifestyle > c.declared_income * 4.0):
                    y_raw[idx] = 1.0
                citizen_mask[idx] = True
        elif node_type == "Vehicle":
            x_raw[idx, 2] = G.nodes[node].get("value", 0.0)
        elif node_type == "Property":
            x_raw[idx, 2] = G.nodes[node].get("value", 0.0)
        elif node_type == "UtilityBill":
            x_raw[idx, 3] = G.nodes[node].get("annual_bill", 0.0)
        elif node_type == "BankAccount":
            x_raw[idx, 2] = G.nodes[node].get("average_balance", 0.0)
            x_raw[idx, 5] = G.nodes[node].get("volume", 0.0)
        elif node_type == "TravelRecord":
            x_raw[idx, 4] = G.nodes[node].get("cost", 0.0)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    
    edges = []
    for u, v in G.edges():
        edges.append([node_to_idx[u], node_to_idx[v]])
        edges.append([node_to_idx[v], node_to_idx[u]])
        
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
    x_tensor = torch.tensor(x_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y_raw, dtype=torch.float).unsqueeze(-1)
    
    citizen_indices = np.where(citizen_mask)[0]
    np.random.shuffle(citizen_indices)
    split = int(len(citizen_indices) * 0.8)
    train_idx = torch.tensor(citizen_indices[:split], dtype=torch.long)
    
    model = TaxGCN(in_features=8, hidden_features=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(40):
        optimizer.zero_grad()
        out = model(x_tensor, edge_index)
        loss = criterion(out[train_idx], y_tensor[train_idx])
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(x_tensor, edge_index).squeeze(-1).numpy()
        
    updated = 0
    for idx, is_cit in enumerate(citizen_mask):
        if is_cit:
            node_name = idx_to_node[idx]
            cnic = node_name.replace("Citizen_", "")
            prob = float(preds[idx])
            c = citizens_db.get(cnic)
            if c:
                c.anomaly_probability = prob
                updated += 1
                
    session.commit()
    print(f"[SUCCESS] GCN training and anomaly inference completed. Updated {updated} citizens.")

# -------------------------------------------------------------
# 5. Risk Scoring & XAI Breakdown with Entity Resolution
# -------------------------------------------------------------
def calculate_compliance_deviation_scores(session, er_engine=None):
    print("Calculating Tax Compliance Deviation Scores (leveraging resolved entities)...")
    citizens = session.query(Citizen).all()
    vehicles = session.query(Vehicle).all()
    properties = session.query(Property).all()
    utility_bills = session.query(UtilityBill).all()
    bank_accounts = session.query(BankAccount).all()
    travels = session.query(TravelRecord).all()
    
    # Pre-map lists based on resolved CNICs to correctly aggregate details
    v_map = {}
    p_map = {}
    u_map = {}
    b_map = {}
    t_map = {}
    
    # Helper to resolve owner or fall back to clean cnic
    def resolve_cnic_owner(raw_cnic, address=""):
        if er_engine:
            resolved, score = er_engine.resolve_citizen(raw_cnic, query_address=address)
            if resolved:
                return clean_cnic(resolved.cnic)
        return clean_cnic(raw_cnic)
        
    # Map assets
    for v in vehicles:
        owner = resolve_cnic_owner(v.owner_cnic)
        v_map.setdefault(owner, []).append(v)
        
    for p in properties:
        owner = resolve_cnic_owner(p.owner_cnic, address=p.address)
        p_map.setdefault(owner, []).append(p)
        
    for u in utility_bills:
        owner = resolve_cnic_owner(u.owner_cnic, address=u.address)
        u_map.setdefault(owner, []).append(u)
        
    for b in bank_accounts:
        owner = resolve_cnic_owner(b.owner_cnic)
        b_map.setdefault(owner, []).append(b)
        
    for t in travels:
        owner = resolve_cnic_owner(t.owner_cnic)
        t_map.setdefault(owner, []).append(t)
        
    for c in citizens:
        c_cnic = clean_cnic(c.cnic)
        
        # 1. Vehicles (Cap: 15,000,000 PKR)
        user_v = v_map.get(c_cnic, [])
        total_v = sum(v.value for v in user_v)
        s_vehicle = min((total_v / 15000000.0) * 100.0, 100.0)
        
        # 2. Properties (Cap: 60,000,000 PKR)
        user_p = p_map.get(c_cnic, [])
        total_p = sum(p.value for p in user_p)
        s_property = min((total_p / 60000000.0) * 100.0, 100.0)
        
        # 3. Utilities (annualized) (Cap: 400,000 PKR)
        user_u = u_map.get(c_cnic, [])
        total_u = sum(u.amount for u in user_u)
        s_utility = min((total_u / 400000.0) * 100.0, 100.0)
        
        # 4. Travel (Cap: 2,000,000 PKR)
        user_t = t_map.get(c_cnic, [])
        total_t = sum(t.cost for t in user_t)
        s_travel = min((total_t / 2000000.0) * 100.0, 100.0)
        
        # 5. Bank average balance (Cap: 20,000,000 PKR)
        user_b = b_map.get(c_cnic, [])
        max_b_bal = max([ba.average_balance for ba in user_b], default=0.0)
        s_bank = min((max_b_bal / 20000000.0) * 100.0, 100.0)
        
        # 6. Declared Income Offset (Cap: 8,000,000 PKR)
        s_income = min((c.declared_income / 8000000.0) * 100.0, 100.0)
        
        # Base non-filer risk (+30.0 score penalty)
        base_filer_risk = 30.0 if c.declared_income == 0 else 0.0
        
        # Calculate upkeep and under-reporting discrepancy penalty
        upkeep = (total_v * 0.10) + (total_p * 0.02) + total_u + total_t
        discrepancy_penalty = 25.0 if upkeep > c.declared_income else 0.0
        
        # Risk Score equation
        raw_score = (
            (0.25 * s_vehicle) +
            (0.30 * s_property) +
            (0.25 * s_utility) +
            (0.20 * s_travel) +
            (0.15 * s_bank) -
            (0.15 * s_income)
        )
        
        # Spread raw score and add base risk + discrepancy penalty
        compliance_score = (raw_score * 1.2) + base_filer_risk + discrepancy_penalty
        compliance_score = max(min(compliance_score, 100.0), 0.0)
        
        if compliance_score < 40.0:
            risk_band = "Low Risk"
        elif compliance_score <= 75.0:
            risk_band = "Medium Risk"
        else:
            risk_band = "High Risk"
            
        c.compliance_score = round(compliance_score, 2)
        c.risk_band = risk_band
        
        # Calculate XAI audit reasons
        reasons = []
        for v in user_v:
            reasons.append(f"Owns vehicle: {v.make} {v.model} ({v.year}) valued at PKR {v.value:,.0f}")
        for p in user_p:
            reasons.append(f"Owns {p.property_type} property in {p.city} | Value: PKR {p.value:,.0f}")
        if total_u > 200000:
            reasons.append(f"High annual utility consumption of PKR {total_u:,.0f}")
        if user_t:
            reasons.append(f"Travelled abroad {len(user_t)} time(s) (Cost: PKR {total_t:,.0f})")
        if max_b_bal > 5000000:
            reasons.append(f"Average banking balance up to PKR {max_b_bal:,.0f}")
            
        reasons.append(f"Declared annual income = PKR {c.declared_income:,.0f}")
        
        upkeep = (total_v * 0.10) + (total_p * 0.02) + total_u + total_t
        if c.declared_income == 0:
            if len(reasons) > 1:
                conclusion = f"Tax non-filer with registered assets/lifestyle footprint. Est. annual upkeep of PKR {upkeep:,.0f} suggests high tax evasion risk."
            else:
                conclusion = "Tax non-filer with minimal formal wealth trail."
        else:
            if upkeep > c.declared_income * 2.0:
                conclusion = f"Tax filer with massive under-reporting. Lifestyle upkeep (est. PKR {upkeep:,.0f}/year) exceeds declared income of PKR {c.declared_income:,.0f} by {upkeep/max(c.declared_income,1):.1f}x."
            else:
                conclusion = "Lifestyle indicators match declared annual income."
                
        explanation_payload = {
            "citizen_name": c.name,
            "cnic": c.cnic,
            "risk_score": c.compliance_score,
            "anomaly_probability": c.anomaly_probability,
            "reasons": reasons,
            "conclusion": conclusion,
            "breakdown": {
                "vehicles_value": total_v,
                "properties_value": total_p,
                "annual_utilities": total_u,
                "annual_travel_cost": total_t,
                "max_bank_balance": max_b_bal,
                "declared_income": c.declared_income
            }
        }
        c.xai_explanations = json.dumps(explanation_payload)
        
    session.commit()
    print("[SUCCESS] Compliance scores and XAI reasons updated.")

# -------------------------------------------------------------
# 6. Sub-Network Visualization
# -------------------------------------------------------------
def save_graph_visualization(session, G):
    print("Generating network visualization for high-risk sub-network...")
    high_risk_citizens = session.query(Citizen).filter(Citizen.risk_band == "High Risk")\
                                .order_by(Citizen.compliance_score.desc()).limit(20).all()
    high_risk_cnics = [c.cnic for c in high_risk_citizens]
    seed_nodes = [f"Citizen_{cnic}" for cnic in high_risk_cnics if G.has_node(f"Citizen_{cnic}")]
    
    if not seed_nodes:
        print("[WARNING] No high-risk citizen nodes found in graph to visualize.")
        return
        
    subgraph_nodes = set(seed_nodes)
    for node in seed_nodes:
        subgraph_nodes.update(G.predecessors(node))
        subgraph_nodes.update(G.successors(node))
        
    sub_G = G.subgraph(subgraph_nodes).copy()
    draw_G = nx.Graph()
    for u, v, data in sub_G.edges(data=True):
        draw_G.add_edge(u, v)
    for n in draw_G.nodes:
        if n in sub_G.nodes:
            draw_G.nodes[n].update(sub_G.nodes[n])
            
    color_map = []
    labels = {}
    node_sizes = []
    
    colors = {
        "Citizen": "#1F77B4",
        "Vehicle": "#FF7F0E",
        "Property": "#2CA02C",
        "UtilityBill": "#D62728",
        "BankAccount": "#9467BD",
        "TravelRecord": "#17BECF"
    }
    
    for node in draw_G.nodes:
        label = draw_G.nodes[node].get("label", "Asset")
        color_map.append(colors.get(label, "#BCBD22"))
        if label == "Citizen":
            labels[node] = draw_G.nodes[node].get("name", node)
            node_sizes.append(300)
        else:
            labels[node] = ""
            node_sizes.append(80)
            
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(draw_G, k=0.15, seed=42)
    
    ax = plt.gca()
    ax.set_facecolor("#121212")
    plt.gcf().patch.set_facecolor("#121212")
    
    nx.draw_networkx_edges(draw_G, pos, alpha=0.3, edge_color="#FFFFFF")
    nx.draw_networkx_nodes(draw_G, pos, node_color=color_map, node_size=node_sizes, alpha=0.95)
    
    citizen_pos = {k: [v[0], v[1] + 0.03] for k, v in pos.items() if k.startswith("Citizen_")}
    citizen_labels = {k: v for k, v in labels.items() if k.startswith("Citizen_")}
    nx.draw_networkx_labels(draw_G, citizen_pos, labels=citizen_labels, font_size=9, 
                            font_color="#FFFFFF", font_weight="bold")
    
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=10, label=lbl) 
                      for lbl, col in colors.items()]
    plt.legend(handles=legend_handles, loc="upper right", facecolor="#1E1E1E", edgecolor="#333333", labelcolor="#FFFFFF")
    plt.title("FBR Tax Intelligence Knowledge Graph (High-Risk Sub-Network)", color="#FFFFFF", fontsize=16, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("tax_knowledge_graph.png", dpi=300, facecolor="#121212")
    plt.close()
    print("[SUCCESS] Sub-network visualization saved as 'tax_knowledge_graph.png'.")

# -------------------------------------------------------------
# 7. Automated Audit Trail
# -------------------------------------------------------------
def generate_audit_trail(session, er_engine=None):
    print("Generating automated text-based audit trail...")
    flagged = session.query(Citizen).order_by(Citizen.compliance_score.desc()).limit(10).all()
    
    vehicles = session.query(Vehicle).all()
    properties = session.query(Property).all()
    utility_bills = session.query(UtilityBill).all()
    bank_accounts = session.query(BankAccount).all()
    travels = session.query(TravelRecord).all()
    
    v_map = {}
    p_map = {}
    u_map = {}
    b_map = {}
    t_map = {}
    
    def resolve_cnic_owner(raw_cnic, address=""):
        if er_engine:
            resolved, score = er_engine.resolve_citizen(raw_cnic, query_address=address)
            if resolved:
                return clean_cnic(resolved.cnic)
        return clean_cnic(raw_cnic)
        
    for v in vehicles:
        owner = resolve_cnic_owner(v.owner_cnic)
        v_map.setdefault(owner, []).append(v)
        
    for p in properties:
        owner = resolve_cnic_owner(p.owner_cnic, address=p.address)
        p_map.setdefault(owner, []).append(p)
        
    for u in utility_bills:
        owner = resolve_cnic_owner(u.owner_cnic, address=u.address)
        u_map.setdefault(owner, []).append(u)
        
    for b in bank_accounts:
        owner = resolve_cnic_owner(b.owner_cnic)
        b_map.setdefault(owner, []).append(b)
        
    for t in travels:
        owner = resolve_cnic_owner(t.owner_cnic)
        t_map.setdefault(owner, []).append(t)
    
    audit_lines = []
    audit_lines.append("=" * 80)
    audit_lines.append("FBR TAX INTELLIGENCE PLATFORM - AUTOMATED AUDIT TRAIL")
    audit_lines.append("Generated on: 2026-06-13")
    audit_lines.append("Targeting top 10 highest compliance deviation profiles.")
    audit_lines.append("=" * 80)
    audit_lines.append("\n")
    
    for idx, c in enumerate(flagged):
        c_cnic = clean_cnic(c.cnic)
        user_v = v_map.get(c_cnic, [])
        user_p = p_map.get(c_cnic, [])
        user_u = u_map.get(c_cnic, [])
        user_b = b_map.get(c_cnic, [])
        user_t = t_map.get(c_cnic, [])
        
        v_value = sum(v.value for v in user_v)
        p_value = sum(p.value for p in user_p)
        annual_u = sum(u.amount for u in user_u)
        t_cost = sum(t.cost for t in user_t)
        max_b = max([ba.average_balance for ba in user_b], default=0.0)
        
        upkeep = (v_value * 0.10) + (p_value * 0.02) + annual_u + t_cost
        
        audit_lines.append(f"CASE PROFILE #{idx+1}: {c.name}")
        audit_lines.append(f"  CNIC:                 {c.cnic}")
        audit_lines.append(f"  City:                 {c.city}")
        audit_lines.append(f"  Risk Band:            {c.risk_band} (Deviation Score: {c.compliance_score:.2f})")
        audit_lines.append(f"  GNN Anomaly Prob:     {c.anomaly_probability:.4f}")
        audit_lines.append(f"  Declared Income:      PKR {c.declared_income:,.2f}")
        audit_lines.append(f"  Tax Paid:             PKR {c.tax_paid:,.2f}")
        audit_lines.append(f"  Est. Annual Upkeep:   PKR {upkeep:,.2f}")
        audit_lines.append("  ASSET INVENTORY & TRANSACTIONS:")
        
        if user_v:
            audit_lines.append("    * Registered Vehicles:")
            for v in user_v:
                audit_lines.append(f"      - {v.make} {v.model} ({v.year}) | Plate: {v.plate_number} | Value: PKR {v.value:,.2f}")
        else:
            audit_lines.append("    * Registered Vehicles: None found")
            
        if user_p:
            audit_lines.append("    * Property Deeds:")
            for p in user_p:
                audit_lines.append(f"      - {p.property_type} in {p.city} ({p.area}) | Address: {p.address} | Value: PKR {p.value:,.2f}")
        else:
            audit_lines.append("    * Property Deeds: None found")
            
        if user_u:
            audit_lines.append(f"    * Utility Bills: {len(user_u)} invoices | Annual Total: PKR {annual_u:,.2f}")
        else:
            audit_lines.append("    * Utility Bills: None found")
            
        if user_b:
            audit_lines.append("    * Formal Banking Portfolios:")
            for b in user_b:
                audit_lines.append(f"      - {b.bank_name} Account: {b.account_number} | Avg Balance: PKR {b.average_balance:,.2f} | Volume: PKR {b.transaction_volume:,.2f}")
        else:
            audit_lines.append("    * Formal Banking Portfolios: None found")
            
        if user_t:
            audit_lines.append("    * International Travel History:")
            for t in user_t:
                audit_lines.append(f"      - Destination: {t.destination} | Date: {t.travel_date} | Cost: PKR {t.cost:,.2f}")
        else:
            audit_lines.append("    * International Travel History: None found")
            
        audit_lines.append("  AUDIT EVALUATION SUMMARY:")
        is_filer = c.declared_income > 0
        if not is_filer:
            audit_lines.append("    [CRITICAL] Subject is a tax NON-FILER despite holding substantial registered assets and a utility footprint.")
            audit_lines.append(f"    Action: Recommend immediate asset-freezing and issuing statutory tax compliance notices. Potential unregistered wealth: PKR {v_value + p_value:,.2f}.")
        else:
            gap_ratio = upkeep / max(c.declared_income, 1.0)
            if gap_ratio > 2.0:
                audit_lines.append(f"    [CRITICAL] Severe wealth-to-income mismatch. Est. annual lifestyle maintenance (PKR {upkeep:,.2f}) exceeds declared income (PKR {c.declared_income:,.2f}) by {gap_ratio:.1f}x.")
                audit_lines.append("    Action: Initiate detailed lifestyle audit.")
            else:
                audit_lines.append("    [NOTICE] Asset footprint matches declared income bands.")
                
        audit_lines.append("-" * 80)
        audit_lines.append("\n")
        
    with open("tax_audit_trail.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))
    print("[SUCCESS] Text-based audit trail saved to 'tax_audit_trail.txt'.")

# -------------------------------------------------------------
# Main Orchestration Execution
# -------------------------------------------------------------
def main():
    print("=============================================================")
    print("STARTING FBR TAX PLATFORM KNOWLEDGE PIPELINE")
    print("=============================================================")
    
    if not os.path.exists(DATABASE_PATH):
        print(f"[ERROR] Database file '{DATABASE_PATH}' not found in current directory.")
        return
        
    session = SessionLocal()
    try:
        # 1. Entity Resolution
        er_engine = EntityResolutionEngine(session)
        run_entity_resolution_evaluation(session, er_engine)
        
        # 2. Build Knowledge Graph
        G = build_knowledge_graph(session, er_engine)
        
        # 3. Compute Graph Centrality
        compute_graph_metrics(session, G)
        
        # 4. Train GNN Anomaly Classifier
        train_gnn(session, G)
        
        # 5. Compute Deviation Score and XAI Breakdown (leveraging ER engine)
        calculate_compliance_deviation_scores(session, er_engine)
        
        # 6. Save Visualization
        save_graph_visualization(session, G)
        
        # 7. Generate Audit Trail (leveraging ER engine)
        generate_audit_trail(session, er_engine)
        
        print("\n=============================================================")
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=============================================================")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    main()
