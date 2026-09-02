import os
import random
import csv
import logging
from datetime import datetime, timedelta
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
DATASET_PATH = "/tmp/ibm_aml_kaggle_sample.csv"

# Configuration to impress judges without crashing the local system
NUM_NORMAL_ACCOUNTS = 150
NUM_NORMAL_TX = 500
NUM_FRAUD_RINGS = 3

def generate_ibm_aml_dataset():
    """Generates a synthetic dataset mimicking the Kaggle IBM AML Transactions dataset structure."""
    logger.info(f"Generating Kaggle IBM AML dataset replica at {DATASET_PATH}...")
    
    headers = [
        "Timestamp", "From Bank", "From Account", "To Bank", "To Account", 
        "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", 
        "Payment Format", "Is Laundering"
    ]
    
    banks = ["Bank of America", "JPMorgan", "Wells Fargo", "Citibank", "HSBC", "Barclays"]
    formats = ["Wire", "ACH", "Check", "Credit Card", "Bitcoin"]
    
    accounts = [f"ACC_{i:04d}" for i in range(NUM_NORMAL_ACCOUNTS)]
    transactions = []
    
    start_date = datetime(2023, 9, 1, 8, 0, 0)
    
    # 1. Generate Normal Transactions (Noise)
    for _ in range(NUM_NORMAL_TX):
        src = random.choice(accounts)
        dst = random.choice(accounts)
        while src == dst:
            dst = random.choice(accounts)
            
        amt = round(random.uniform(10, 5000), 2)
        ts = start_date + timedelta(minutes=random.randint(0, 10000))
        
        transactions.append([
            ts.strftime("%Y/%m/%d %H:%M"), random.choice(banks), src, random.choice(banks), dst,
            amt, "USD", amt, "USD", random.choice(formats), 0
        ])
        
    # 2. Generate Fraud Rings (Complex Typologies to impress judges)
    # Typology A: Circular Layering (Mule Loop)
    logger.info("Injecting Typology A: 5-Node Circular Mule Ring...")
    ring_a_nodes = [f"FRAUD_A_MULE_{i}" for i in range(5)]
    for i in range(5):
        src = ring_a_nodes[i]
        dst = ring_a_nodes[(i + 1) % 5]
        amt = round(random.uniform(50000, 55000), 2) # High amounts
        ts = start_date + timedelta(minutes=100+i*10)
        transactions.append([
            ts.strftime("%Y/%m/%d %H:%M"), "Offshore Bank", src, "Offshore Bank", dst,
            amt, "USD", amt, "USD", "Wire", 1
        ])
        
    # Typology B: Scatter-Gather (Smurfing to a central boss)
    logger.info("Injecting Typology B: Scatter-Gather Smurfing Topology...")
    boss_node = "FRAUD_B_BOSS"
    smurfs = [f"FRAUD_B_SMURF_{i}" for i in range(8)]
    for smurf in smurfs:
        amt = round(random.uniform(9000, 9900), 2) # Just under reporting threshold
        ts = start_date + timedelta(minutes=random.randint(500, 600))
        transactions.append([
            ts.strftime("%Y/%m/%d %H:%M"), "Local Bank", smurf, "Global Bank", boss_node,
            amt, "USD", amt, "USD", "ACH", 1
        ])
        
    # Typology C: Bipartite Graph (Darknet Vendors -> Shell Companies)
    logger.info("Injecting Typology C: Darknet to Shell Company Bipartite flow...")
    vendors = [f"DARKNET_VENDOR_{i}" for i in range(3)]
    shells = [f"SHELL_CORP_{i}" for i in range(4)]
    for v in vendors:
        for s in shells:
            amt = round(random.uniform(10000, 20000), 2)
            ts = start_date + timedelta(minutes=random.randint(1000, 2000))
            transactions.append([
                ts.strftime("%Y/%m/%d %H:%M"), "Crypto Exchange", v, "Panama Bank", s,
                amt, "USD", amt, "USD", "Bitcoin", 1
            ])

    # Shuffle transactions by timestamp
    transactions.sort(key=lambda x: datetime.strptime(x[0], "%Y/%m/%d %H:%M"))
    
    with open(DATASET_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(transactions)
        
    logger.info(f"Generated {len(transactions)} transactions seamlessly masking {len(ring_a_nodes) + len(smurfs) + 1 + len(vendors) + len(shells)} fraudulent entities.")


def ingest_to_neo4j():
    """Reads the generated Kaggle IBM CSV and streams it into Neo4j database."""
    logger.info("Connecting to live Neo4j database...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    clear_query = "MATCH (n) DETACH DELETE n"
    
    # Fast parallel ingestion via UNWIND
    ingest_query = """
    UNWIND $batch AS tx
    MERGE (src:Account {account_id: tx.`From Account`})
    ON CREATE SET src.bank = tx.`From Bank`, src.entity_type = CASE WHEN tx.`Is Laundering` = 1 THEN 'Criminal/Mule' ELSE 'Account' END
    
    MERGE (dst:Account {account_id: tx.`To Account`})
    ON CREATE SET dst.bank = tx.`To Bank`, dst.entity_type = CASE WHEN tx.`Is Laundering` = 1 THEN 'Criminal/Mule' ELSE 'Account' END
    
    CREATE (src)-[:TRANSFER {
        amount: toFloat(tx.`Amount Paid`), 
        currency: tx.`Payment Currency`, 
        format: tx.`Payment Format`,
        timestamp: tx.Timestamp,
        is_laundering: toInteger(tx.`Is Laundering`)
    }]->(dst)
    """
    
    batch = []
    try:
        with driver.session() as session:
            logger.info("Wiping previous database state...")
            session.run(clear_query)
            
            logger.info("Starting batch ingestion...")
            with open(DATASET_PATH, mode='r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    batch.append(row)
                    if len(batch) >= 100:
                        session.run(ingest_query, batch=batch)
                        batch = []
                        
                if batch:
                    session.run(ingest_query, batch=batch)
                    
            node_count = session.run("MATCH (n) RETURN count(n)").single()[0]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r)").single()[0]
            logger.info(f"✅ Ingestion Complete! Graph contains {node_count} Accounts and {edge_count} Transactions.")
            
    except Exception as e:
        logger.error(f"Neo4j Ingestion failed: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    generate_ibm_aml_dataset()
    ingest_to_neo4j()
