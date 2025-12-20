from urllib.parse import quote_plus

def generate_maps_url(name: str, address: str, postcode: str = None) -> str:
    """
    Generates a Google Maps search URL for a given establishment.
    
    Args:
        name: Name of the restaurant.
        address: Address line.
        postcode: Postcode (optional).
        
    Returns:
        A Google Maps search URL string.
    """
    query_parts = [name, address]
    if postcode:
        query_parts.append(postcode)
    
    query_string = " ".join(query_parts)
    encoded_query = quote_plus(query_string)
    
    return f"https://www.google.com/maps/search/?api=1&query={encoded_query}"
