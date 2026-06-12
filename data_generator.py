import sqlite3
import random

def generate_vanguard_big_data():
    conn = sqlite3.connect('tax_graph_ai.db')
    cursor = conn.cursor()
    
    # 1. Drop existing table to avoid old static data conflict
    cursor.execute("DROP TABLE IF EXISTS tax_compliance_summary")
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tax_compliance_summary (
            name TEXT, 
            cnic TEXT, 
            reported_income INT, 
            utility_bills INT, 
            asset_score TEXT, 
            risk_tier TEXT
        )
    """)
    
    # Base components to generate 1000+ unique Pakistani names programmatically
    first_names = ['Adeel', 'Kamran', 'Zain', 'Asif', 'Hamza', 'Bilal', 'Usman', 'Faisal', 'Omar', 'Ali', 'Zeeshan', 'Sajid', 'Arsalan', 'Haris', 'Tariq', 'Nabeel', 'Waqas', 'Junaid', 'Raza', 'Farhan']
    last_names = ['Haider', 'Khan', 'Ahmed', 'Shah', 'Malik', 'Butt', 'Chaudhry', 'Javed', 'Siddiqui', 'Riaz', 'Bajwa', 'Abbasi', 'Qureshi', 'Ghuman', 'Zafar', 'Iqbal', 'Ansari', 'Latif', 'Aziz', 'Mahmood']
    
    bulk_data = []
    
    # Hardcoded VIP Special Test Cases for instant live demo checks
    vip_profiles = [
        ('Adeel Haider', '35202-1111111-1', 15000, 350000, '2500cc SUV', 'HIGH RISK'),
        ('Adil Haidar', '35202-1111111-1', 15000, 350000, '2500cc SUV', 'HIGH RISK'), # Entity Resolution Variant
        ('Kamran Khan', '35202-2222222-2', 75000, 95000, '1300cc Sedan', 'MEDIUM RISK'),
        ('Zain Ahmed', '35202-3333333-3', 450000, 20000, '1000cc Hatchback', 'LOW RISK')
    ]
    bulk_data.extend(vip_profiles)
    
    # 2. Loop to generate 1025 records with mixed tier distributions
    random.seed(42) # For consistent generation
    
    for i in range(1025):
        f_name = random.choice(first_names)
        l_name = random.choice(last_names)
        full_name = f"{f_name} {l_name}"
        
        # Avoid duplicating the exact key names we want to search manually
        if full_name in ['Adeel Haider', 'Kamran Khan', 'Zain Ahmed']:
            full_name = f"{f_name} {random.choice(last_names)} {i}"
            
        cnic = f"35202-{random.randint(1000000, 9999999)}-{random.randint(1, 9)}"
        
        # Distribute risks strategically (Mix percentage: 25% High, 35% Medium, 40% Low)
        roll = random.random()
        
        if roll < 0.25:
            # HIGH RISK: Low income reported, huge luxury bills & assets
            income = random.randint(0, 30000)
            bills = random.randint(150000, 450000)
            asset = random.choice(['2000cc SUV', '4000cc Cruiser', 'Commercial Property', 'Luxury Mansion'])
            tier = 'HIGH RISK'
        elif roll < 0.60:
            # MEDIUM RISK: Average income but lifestyle metrics slightly cross limits
            income = random.randint(50000, 90000)
            bills = random.randint(65000, 120000)
            asset = random.choice(['1300cc Sedan', '1500cc Sedan', 'Residential Plot'])
            tier = 'MEDIUM RISK'
        else:
            # LOW RISK: High income perfectly matching clean lifestyle metrics
            income = random.randint(250000, 800000)
            bills = random.randint(15000, 40000)
            asset = random.choice(['1000cc Hatchback', 'Bike Only', 'No Registered Vehicle'])
            tier = 'LOW RISK'
            
        bulk_data.append((full_name, cnic, income, bills, asset, tier))
        
    # 3. Bulk Insert into SQLite for extreme runtime efficiency
    cursor.executemany("INSERT INTO tax_compliance_summary VALUES (?,?,?,?,?,?)", bulk_data)
    conn.commit()
    
    print(f"[Vanguard Engine Status] Database initialized successfully.")
    print(f"Total structured multi-tier records active: {len(bulk_data)}")
    conn.close()

if __name__ == "__main__":
    generate_vanguard_big_data()
