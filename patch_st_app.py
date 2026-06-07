import sys

with open('app/ui/st_app.py', 'r') as f:
    content = f.read()

# Replace block 1
content = content.replace(
    '            # 2. Update Status\n            with col_update:\n                st.write("Update Manual Review Status & Rating:")\n                user_rating = st.slider("User Rating (1-10)", min_value=1, max_value=10, value=5, step=1)\n                c_acc, c_rej, c_pend, c_reset = st.columns(4)',
    '            # 2. Update Status\n            with col_update:\n                st.write("Update Manual Review Status:")\n                c_acc, c_rej, c_pend, c_reset = st.columns(4)'
)

# Replace block 2
content = content.replace(
    '                        rating_to_save = user_rating if new_status != "not reviewed" else None\n                        \n                        # Prepare update DataFrame\n                        df_updates = pd.DataFrame({\n                            \'fhrsid\': ids_to_update,\n                            \'manual_review\': [new_status] * len(ids_to_update),\n                            \'user_rating\': [rating_to_save] * len(ids_to_update)\n                        })',
    '                        # Prepare update DataFrame\n                        df_updates = pd.DataFrame({\n                            \'fhrsid\': ids_to_update,\n                            \'manual_review\': [new_status] * len(ids_to_update)\n                        })'
)

# Replace block 3
old_block_3 = """                                         if 'manual_review' in df_s.columns:
                                             st.session_state.df_enriched.loc[mask, 'manual_review'] = new_status
                                         if 'user_rating' in df_s.columns:
                                             st.session_state.df_enriched.loc[mask, 'user_rating'] = rating_to_save
                                         elif rating_to_save is not None:
                                             st.session_state.df_enriched['user_rating'] = None
                                             st.session_state.df_enriched.loc[mask, 'user_rating'] = rating_to_save
                                
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Update failed: {msg}")
                    else:
                         st.error("Could not identify FHRSID for updates.")"""

new_block_3 = """                                         if 'manual_review' in df_s.columns:
                                             st.session_state.df_enriched.loc[mask, 'manual_review'] = new_status
                                
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Update failed: {msg}")
                    else:
                         st.error("Could not identify FHRSID for updates.")
        
            # 3. Individual User Rating (Only if 1 row selected)
            if len(selected_rows) == 1:
                st.divider()
                st.write("### ⭐ Rate Restaurant")
                row = selected_rows.iloc[0]
                col_map = {c.lower(): c for c in row.index}
                id_col = col_map.get('fhrsid')
                biz_name = row.get('BusinessName') or row.get('businessname') or "Unknown Restaurant"

                current_rating = row.get('user_rating')
                if pd.isna(current_rating):
                    current_rating = 5

                user_rating = st.slider(f"Your Rating for {biz_name} (1-10)", min_value=1, max_value=10, value=int(current_rating), step=1)
                if st.button("Submit Rating", type="primary"):
                    if id_col:
                        fhrsid_to_update = str(row[id_col])
                        with st.spinner("Saving rating..."):
                            from app.services.bq_utils import update_rows_in_bigquery
                            success = update_rows_in_bigquery(
                                project_id, dataset_id, table_id,
                                fhrsid_to_update,
                                {'user_rating': user_rating}
                            )
                            if success:
                                st.success("Rating saved!")
                                if 'df_enriched' in st.session_state:
                                    df_s = st.session_state.df_enriched
                                    s_col_map = {c.lower(): c for c in df_s.columns}
                                    s_id_col = s_col_map.get('fhrsid')
                                    if s_id_col:
                                        mask = df_s[s_id_col].astype(str) == fhrsid_to_update
                                        if 'user_rating' not in df_s.columns:
                                            st.session_state.df_enriched['user_rating'] = None

                                        st.session_state.df_enriched.loc[mask, 'user_rating'] = user_rating
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to save rating.")
                    else:
                        st.error("Could not identify FHRSID.")"""

content = content.replace(old_block_3, new_block_3)

with open('app/ui/st_app.py', 'w') as f:
    f.write(content)

