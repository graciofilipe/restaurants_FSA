import streamlit as st
import pandas as pd
from app.services.bq_utils import bulk_update_reviews

def render_bulk_update_ui(project_id, dataset_id, table_id, selected_rows):
    """
    Renders the UI for bulk updating review status of selected rows.
    """
    st.subheader("Bulk Update Status")
    
    if not selected_rows:
        st.info("Select rows in the table above to perform bulk update.")
        st.button("Update Status for Selected Rows", disabled=True)
        return

    st.write(f"Selected {len(selected_rows)} restaurants.")
    
    status_options = ["pending", "accepted", "rejected", "not reviewed"]
    new_status = st.selectbox("Select New Status", options=status_options)
    
    if st.button("Update Status for Selected Rows"):
        # Construct DataFrame
        fhrsids = [str(row.get('fhrsid')) for row in selected_rows if row.get('fhrsid')]
        
        if not fhrsids:
            st.error("No valid FHRSIDs found in selection.")
            return

        df_update = pd.DataFrame({
            'fhrsid': fhrsids,
            'manual_review': [new_status] * len(fhrsids)
        })
        
        with st.spinner(f"Updating {len(fhrsids)} records..."):
            success, message = bulk_update_reviews(project_id, dataset_id, table_id, df_update)
            
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(f"Update failed: {message}")
