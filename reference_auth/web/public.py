import os
from flask import Blueprint, request, render_template, url_for, session, redirect
from services.storage import StorageService

public_bp = Blueprint('public', __name__)

@public_bp.route("/", methods=["GET"])
def index():
    # Enforce Authentication
    if not session.get('user'):
        return redirect(url_for('auth.login_page'))

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        return "GOOGLE_CLOUD_PROJECT not set", 500
        
    storage = StorageService(project_id)
    query = request.args.get("query", "")
    
    # Facet Handling
    raw_facet_keys = request.args.getlist("facet_key")
    facet_keys = [k for k in raw_facet_keys if k.strip()]
    if not facet_keys: 
        facet_keys = ["episode_year_month", "podcast_name"]
    
    # Pagination
    page_token = request.args.get("page_token")
    page_size = 10

    # Filtering
    filter_values = request.args.getlist("filter")
    filter_key = request.args.get("filter_key")
    filter_str = None
    
    # Default target key if not provided
    target_key = filter_key if filter_key else facet_keys[0]

    if filter_values:
        conditions = []
        for val in filter_values:
            condition = None
            # Special logic for date range filtering on episode_year_month
            if target_key == "episode_year_month" and len(val) == 7:
                try:
                    from datetime import datetime
                    year, month = map(int, val.split('-'))
                    start_date = datetime(year, month, 1)
                    if month == 12:
                        end_date = datetime(year + 1, 1, 1)
                    else:
                        end_date = datetime(year, month + 1, 1)
                    
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date.strftime("%Y-%m-%d")
                    condition = f'(episode_date >= "{start_str}" AND episode_date < "{end_str}")'
                except ValueError:
                    pass 
            
            if not condition:
                if ":" in val:
                    condition = val
                else:
                    # Construct standard filter: key: "value"
                    condition = f'{target_key}: "{val}"'
            
            if condition:
                conditions.append(condition)
        
        if conditions:
            filter_str = " OR ".join(conditions)

    search_results = None
    facets = None
    next_page_token = None
    
    if query:
        try:
            # 1. Main Search (Filtered) for Results
            response = storage.search(
                query=query, 
                facet_key=facet_keys, 
                page_size=page_size, 
                page_token=page_token,
                filter_str=filter_str
            )
            search_results = response.results
            next_page_token = response.next_page_token
            
            # 2. Facet Search (Unfiltered) for Facet Options
            # If we have filters applied, the main response will have restricted facets.
            # We want to show ALL facet options for the query, as if no filter was applied.
            if filter_str:
                facet_response = storage.search(
                    query=query,
                    facet_key=facet_keys,
                    page_size=0, # We only need facets, not hits
                    filter_str=None
                )
                facets = facet_response.facets
            else:
                facets = response.facets
            
        except Exception as e:
            return f"Error during search: {e}", 500

    # Render the ROBUST template (search.html) but served from root
    return render_template("search.html", 
                           query=query, 
                           facet_keys=facet_keys, 
                           search_results=search_results, 
                           facets=facets, 
                           next_page_token=next_page_token, 
                           selected_filters=filter_values, 
                           filter_key=target_key,
                           user=session.get('user'))
