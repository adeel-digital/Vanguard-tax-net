import os
import sqlite3
import random
import string

DATABASE_PATH = "tax_graph_ai.db"

# Pakistani names and assets details
FIRST_NAMES = ["Muhammad", "Ahmad", "Ali", "Fatima", "Aisha", "Usman", "Hamza", "Zainab", "Bilal", "Omer", "Asif", "Imran", "Kamran", "Muneeb", "Saad", "Zeeshan", "Faisal", "Yasir", "Tariq", "Javed"]
LAST_NAMES = ["Khan", "Ahmed", "Ali", "Mahmood", "Hussain", "Shah", "Iqbal", "Butt", "Chaudhry", "Malik", "Sheikh", "Jatoi", "Qureshi", "Siddiqui"]
CITIES = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar", "Quetta", "Faisalabad", "Multan", "Sialkot", "Gujranwala"]
PROP_TYPES = ["Residential", "Commercial"]
AREAS = ["5 Marla", "10 Marla", "1 Kanal", "120 SqYd", "240 SqYd"]
VEHICLES_MODELS = [("Toyota", "Corolla", 6500000), ("Honda", "Civic", 8000000), ("Suzuki", "Alto", 2800000), ("Suzuki", "Cultus", 4000000), ("Kia", "Sportage", 8500000), ("Toyota", "Prado", 22000000)]
BANKS = ["HBL", "UBL", "MCB", "Meezan Bank", "Bank Alfalah"]
DESTINATIONS = ["Dubai", "London", "Jeddah", "New York", "Istanbul", "Singapore"]

def generate_cnic():
    return f"{random.randint(10000, 99999)}-{random.randint(1000000, 9999999)}-{random.randint(0, 9)}"

def generate_phone():
    return f"0300-{random.randint(1000000, 9999999)}"

def generate_address(city):
    return f"House No. {random.randint(1, 1000)}, Street {random.randint(1, 50)}, Sector G-11, {city}"

def populate_large_db():
    print("=" * 60)
    print("FBR PLATFORM - SCALING DATABASE TO 10 LAKH (1,000,000) TAXPAYERS")
    print("=" * 60)
    
    # Establish raw connection to bypass SQLAlchemy ORM overhead
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Enable SQLite high-performance settings (PRAGMAs)
    print("Optimizing SQLite write buffers...")
    cursor.execute("PRAGMA journal_mode = OFF;")
    cursor.execute("PRAGMA synchronous = OFF;")
    cursor.execute("PRAGMA cache_size = -2000000;") # Use 2GB memory cache
    cursor.execute("PRAGMA locking_mode = EXCLUSIVE;")
    
    # Wiping existing tables to populate clean scaled data
    print("Clearing existing database tables...")
    cursor.execute("DELETE FROM citizens;")
    cursor.execute("DELETE FROM vehicles;")
    cursor.execute("DELETE FROM properties;")
    cursor.execute("DELETE FROM utility_bills;")
    cursor.execute("DELETE FROM bank_accounts;")
    cursor.execute("DELETE FROM travel_records;")
    cursor.execute("DELETE FROM entity_resolution_truth;")
    conn.commit()
    
    # Generation details
    NUM_CITIZENS = 1000000
    BATCH_SIZE = 50000
    
    print(f"\n1. Ingesting {NUM_CITIZENS:,} Citizens...")
    
    citizens_data = []
    all_cnics = []
    
    for i in range(1, NUM_CITIZENS + 1):
        cnic = generate_cnic()
        all_cnics.append(cnic)
        
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        city = random.choice(CITIES)
        address = generate_address(city)
        phone = generate_phone()
        email = f"{name.lower().replace(' ', '')}{random.randint(10,99)}@gmail.com"
        
        # Risk segmentation (70% Low, 20% Medium, 10% High Risk)
        r = random.random()
        if r < 0.10: # High Risk
            declared_inc = 0.0 # Non-filer
            tax_paid = 0.0
            compliance_score = random.uniform(61.0, 95.0)
            risk_band = "High Risk"
        elif r < 0.30: # Medium Risk
            declared_inc = random.randint(300000, 800000)
            tax_paid = declared_inc * 0.05
            compliance_score = random.uniform(31.0, 60.0)
            risk_band = "Medium Risk"
        else: # Low Risk
            declared_inc = random.randint(1200000, 6000000)
            tax_paid = declared_inc * 0.15
            compliance_score = random.uniform(5.0, 30.0)
            risk_band = "Low Risk"
            
        citizens_data.append((
            i, cnic, name, phone, email, city, address, 
            declared_inc, tax_paid, compliance_score, risk_band,
            0.0, "", 0.0, 0.0, 0, "2026-06-13 00:00:00"
        ))
        
        # Batch insert to prevent memory overflow
        if len(citizens_data) >= BATCH_SIZE:
            cursor.executemany("INSERT INTO citizens VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", citizens_data)
            citizens_data = []
            print(f"  - Ingested {i:,} / {NUM_CITIZENS:,} citizens...")
            
    if citizens_data:
        cursor.executemany("INSERT INTO citizens VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", citizens_data)
        
    print("[SUCCESS] 1,000,000 Citizens successfully ingested.")
    
    # 2. Vehicles Registry Ingestion (Approx 300,000 records)
    print("\n2. Ingesting 300,000 Vehicle records...")
    vehicles_data = []
    for v_idx in range(1, 300001):
        owner = random.choice(all_cnics)
        make, model, val = random.choice(VEHICLES_MODELS)
        plate = f"ICT-{random.randint(100, 999)}-{random.choice(string.ascii_uppercase)}"
        year = random.randint(2015, 2026)
        
        # Mangle CNIC for 10% of vehicle owner keys (dashes removed)
        if random.random() < 0.1:
            owner = owner.replace("-", "")
            
        vehicles_data.append((v_idx, plate, owner, make, model, year, float(val)))
        
        if len(vehicles_data) >= BATCH_SIZE:
            cursor.executemany("INSERT INTO vehicles VALUES (?,?,?,?,?,?,?);", vehicles_data)
            vehicles_data = []
            
    if vehicles_data:
        cursor.executemany("INSERT INTO vehicles VALUES (?,?,?,?,?,?,?);", vehicles_data)
    print("[SUCCESS] 300,000 Vehicles successfully ingested.")
    
    # 3. Properties Ingestion (Approx 200,000 records)
    print("\n3. Ingesting 200,000 Property deeds...")
    properties_data = []
    for p_idx in range(1, 200001):
        owner = random.choice(all_cnics)
        city = random.choice(CITIES)
        address = generate_address(city)
        p_type = random.choice(PROP_TYPES)
        area = random.choice(AREAS)
        val = random.randint(5000000, 60000000)
        
        # Mangle CNIC for 10% of property owner keys (alter last digit)
        if random.random() < 0.1:
            owner = owner[:-1] + str((int(owner[-1]) + 1) % 10)
            
        properties_data.append((p_idx, owner, address, city, p_type, area, float(val)))
        
        if len(properties_data) >= BATCH_SIZE:
            cursor.executemany("INSERT INTO properties VALUES (?,?,?,?,?,?,?);", properties_data)
            properties_data = []
            
    if properties_data:
        cursor.executemany("INSERT INTO properties VALUES (?,?,?,?,?,?,?);", properties_data)
    print("[SUCCESS] 200,000 Properties successfully ingested.")
    
    # 4. Bank Accounts Ingestion (Approx 400,000 records)
    print("\n4. Ingesting 400,000 Bank Account portfolios...")
    bank_data = []
    for b_idx in range(1, 400001):
        owner = random.choice(all_cnics)
        bank_name = random.choice(BANKS)
        acc_num = f"PK{random.randint(10,99)}{bank_name[:3].upper()}{random.randint(100000000, 999999999)}"
        avg_bal = random.randint(10000, 15000000)
        tx_vol = avg_bal * random.uniform(1.5, 4.0)
        cash_wd = tx_vol * random.uniform(0.1, 0.3)
        intl_tf = 0.0 if random.random() > 0.1 else random.randint(50000, 500000)
        
        bank_data.append((b_idx, acc_num, owner, bank_name, float(avg_bal), float(tx_vol), float(cash_wd), float(intl_tf)))
        
        if len(bank_data) >= BATCH_SIZE:
            cursor.executemany("INSERT INTO bank_accounts VALUES (?,?,?,?,?,?,?,?);", bank_data)
            bank_data = []
            
    if bank_data:
        cursor.executemany("INSERT INTO bank_accounts VALUES (?,?,?,?,?,?,?,?);", bank_data)
    print("[SUCCESS] 400,000 Bank Accounts successfully ingested.")
    
    # 5. Travel Records Ingestion (Approx 300,000 records)
    print("\n5. Ingesting 300,000 Travel records...")
    travel_data = []
    for t_idx in range(1, 300001):
        owner = random.choice(all_cnics)
        passport = f"PK{random.randint(1000000, 9999999)}"
        dest = random.choice(DESTINATIONS)
        cost = random.randint(100000, 1200000)
        t_date = f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        t_type = random.choice(["Leisure", "Business", "Religious"])
        
        travel_data.append((t_idx, passport, owner, t_date, dest, t_type, float(cost)))
        
        if len(travel_data) >= BATCH_SIZE:
            cursor.executemany("INSERT INTO travel_records VALUES (?,?,?,?,?,?,?);", travel_data)
            travel_data = []
            
    if travel_data:
        cursor.executemany("INSERT INTO travel_records VALUES (?,?,?,?,?,?,?);", travel_data)
    print("[SUCCESS] 300,000 Travel Records successfully ingested.")

    # 6. Utility Bills Ingestion (Approx 1,000,000 records)
    # We ingest 1 bill per meter for 100,000 consumers for 10 months to keep database size optimal
    print("\n6. Ingesting 1,000,000 Utility Bills (100k consumer meters, 10 billing months each)...")
    utility_data = []
    billing_months = [f"2025-{m:02d}" for m in range(3, 13)]
    
    # Create consumer meters
    consumers = []
    for c_idx in range(100000):
        owner = random.choice(all_cnics)
        city = random.choice(CITIES)
        addr = generate_address(city)
        cons_num = f"E{random.randint(100000000, 999999999)}"
        consumers.append((cons_num, owner, addr))
        
    u_bill_idx = 1
    for cons_num, owner, addr in consumers:
        base_bill = random.randint(2000, 40000)
        for month in billing_months:
            bill_amt = base_bill * random.uniform(0.8, 1.2)
            utility_data.append((u_bill_idx, "Electricity", cons_num, owner, addr, float(bill_amt), 1, month))
            u_bill_idx += 1
            
            if len(utility_data) >= BATCH_SIZE:
                cursor.executemany("INSERT INTO utility_bills VALUES (?,?,?,?,?,?,?,?);", utility_data)
                utility_data = []
                
    if utility_data:
        cursor.executemany("INSERT INTO utility_bills VALUES (?,?,?,?,?,?,?,?);", utility_data)
    print("[SUCCESS] 1,000,000 Utility Bills successfully ingested.")
    
    # Final Commit
    print("\nSaving database indexes and structure...")
    conn.commit()
    conn.close()
    
    db_size = os.path.getsize(DATABASE_PATH) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"DATABASE INGESTION COMPLETED SUCCESSFULLY!")
    print(f"Total Database Size: {db_size:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    populate_large_db()
