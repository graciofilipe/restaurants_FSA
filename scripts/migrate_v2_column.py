from google.cloud import bigquery
import argparse

def add_v2_column(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    table = client.get_table(table_ref)
    
    # Check if column already exists
    if any(field.name == 'gemini_insights_structured' for field in table.schema):
        print(f"Column 'gemini_insights_structured' already exists in {table_ref}.")
        return

    # Create new schema definition
    new_schema = table.schema[:]
    new_schema.append(bigquery.SchemaField("gemini_insights_structured", "STRING", mode="NULLABLE"))
    
    table.schema = new_schema
    client.update_table(table, ["schema"])
    print(f"Successfully added 'gemini_insights_structured' to {table_ref}.")

if __name__ == "__main__":
    # You can hardcode these or pass them as args. 
    # Based on your files, I'll set defaults but allow overrides.
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", default="my-project-testing-426113") 
    parser.add_argument("--dataset_id", default="tussell_intelligence_eu") # Adjust if different
    parser.add_argument("--table_id", default="restaurants") # Adjust if different
    
    args = parser.parse_args()
    
    try:
        add_v2_column(args.project_id, args.dataset_id, args.table_id)
    except Exception as e:
        print(f"Migration Failed: {e}")
