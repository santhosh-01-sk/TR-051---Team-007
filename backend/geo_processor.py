"""
Geo-Referenced Post-Processor
Converts pixel-coordinate damage masks into real-world GeoJSON
by mapping pixel positions back to Lat/Lon based on the original Bounding Box.
Produces GeoJSON consumable by Leaflet.js, Google Earth, ArcGIS, and QGIS.
"""

import numpy as np
import cv2
import json


def pixel_to_latlng(x, y, bbox, img_width, img_height):
    """
    Map pixel coordinates (x, y) to real-world (longitude, latitude).
    
    Args:
        x, y: pixel position
        bbox: [west, south, east, north] in decimal degrees
        img_width, img_height: dimensions of the mask image
    
    Returns:
        [longitude, latitude]
    """
    west, south, east, north = bbox
    lon = west + (x / img_width) * (east - west)
    lat = north - (y / img_height) * (north - south)
    return [round(lon, 6), round(lat, 6)]


def mask_to_geojson(seg_mask, damage_mask, bbox):
    """
    Convert damage mask arrays into a geo-referenced GeoJSON FeatureCollection.
    
    Each polygon includes:
      - 'severity': damage class name
      - 'severity_code': numeric damage level
      - 'area_sq_m': approximate area based on bbox extent
    
    Args:
        seg_mask: (H, W) binary numpy array (building footprint)
        damage_mask: (H, W) integer numpy array (0=background, 1=Minor, 2=Major, 3=Destroyed)
        bbox: [west, south, east, north] in decimal degrees
    
    Returns:
        str: Pretty-printed GeoJSON string
    """
    h, w = damage_mask.shape
    west, south, east, north = bbox
    
    # Calculate approximate pixel area in sq meters using the Haversine-based bbox extent
    bbox_width_m = _haversine_distance(south, west, south, east)
    bbox_height_m = _haversine_distance(south, west, north, west)
    pixel_area_sq_m = (bbox_width_m / w) * (bbox_height_m / h)
    
    features = []
    damage_classes = {
        0: {"name": "No Damage", "color": "#22C55E"},
        1: {"name": "Minor",     "color": "#EAB308"},
        2: {"name": "Major",     "color": "#F97316"},
        3: {"name": "Destroyed", "color": "#EF4444"}
    }
    
    for dmg_val, dmg_info in damage_classes.items():
        if dmg_val == 0:
            binary_mask = ((seg_mask > 0) & (damage_mask == 0)).astype(np.uint8)
        else:
            binary_mask = (damage_mask == dmg_val).astype(np.uint8)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area_pixels = cv2.contourArea(cnt)
            if area_pixels < 10:
                continue
            
            # Simplify contour for smaller file size
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Convert pixel coords → real-world lat/lon
            coords = []
            for pt in approx:
                px, py = int(pt[0][0]), int(pt[0][1])
                coords.append(pixel_to_latlng(px, py, bbox, w, h))
            
            if len(coords) >= 3:
                coords.append(coords[0])  # Close the polygon ring
                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "severity": dmg_info["name"],
                        "severity_code": dmg_val,
                        "color": dmg_info["color"],
                        "area_sq_m": round(area_pixels * pixel_area_sq_m, 2),
                        "area_ha": round(area_pixels * pixel_area_sq_m * 0.0001, 4)
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}
        },
        "properties": {
            "bbox": bbox,
            "total_features": len(features),
            "generator": "DisasterAI Pipeline v1.0"
        },
        "features": features
    }
    
    return json.dumps(geojson, indent=2)


def compute_geo_stats(seg_mask, damage_mask, bbox):
    """
    Compute area statistics using real-world measurements from the bounding box.
    
    Returns:
        dict with objects_counted, impact_area_ha, impact_area_sq_km per class
    """
    h, w = damage_mask.shape
    west, south, east, north = bbox
    
    bbox_width_m = _haversine_distance(south, west, south, east)
    bbox_height_m = _haversine_distance(south, west, north, west)
    pixel_area_sq_m = (bbox_width_m / w) * (bbox_height_m / h)
    
    stats = {
        'objects_counted': {'No Damage': 0, 'Minor': 0, 'Major': 0, 'Destroyed': 0},
        'impact_area_ha': {'No Damage': 0.0, 'Minor': 0.0, 'Major': 0.0, 'Destroyed': 0.0},
        'impact_area_sq_km': {'No Damage': 0.0, 'Minor': 0.0, 'Major': 0.0, 'Destroyed': 0.0}
    }
    
    classes = {0: 'No Damage', 1: 'Minor', 2: 'Major', 3: 'Destroyed'}
    for dmg_val, dmg_name in classes.items():
        if dmg_val == 0:
            binary_mask = ((seg_mask > 0) & (damage_mask == 0)).astype(np.uint8)
        else:
            binary_mask = (damage_mask == dmg_val).astype(np.uint8)
        
        num_labels, _ = cv2.connectedComponents(binary_mask)
        stats['objects_counted'][dmg_name] = max(0, num_labels - 1)
        
        pixel_count = int(np.sum(binary_mask))
        area_sq_m = pixel_count * pixel_area_sq_m
        stats['impact_area_ha'][dmg_name] = round(area_sq_m * 0.0001, 3)
        stats['impact_area_sq_km'][dmg_name] = round(area_sq_m * 1e-6, 4)
    
    return stats


def _haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points using Haversine formula."""
    import math
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c
