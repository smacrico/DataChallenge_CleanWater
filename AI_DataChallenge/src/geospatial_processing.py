"""
Geospatial data processing functions for DEM, land cover, and raster sampling.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import warnings

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.windows import from_bounds as window_from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn("rasterio not installed. Geospatial functions will be limited.")

try:
    from scipy.ndimage import sobel
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def sample_raster_at_points(
    raster_path: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    band: int = 1
) -> np.ndarray:
    """
    Sample raster values at given lat/lon coordinates.

    Args:
        raster_path: Path to raster file (GeoTIFF)
        latitudes: Array of latitude values
        longitudes: Array of longitude values
        band: Raster band number to sample (default: 1)

    Returns:
        Array of sampled values
    """
    if not RASTERIO_AVAILABLE:
        print("Warning: rasterio not available. Returning zeros.")
        return np.zeros(len(latitudes))

    try:
        with rasterio.open(raster_path) as src:
            # Get the raster data
            data = src.read(band)

            # Sample at each coordinate
            sampled_values = []
            for lat, lon in zip(latitudes, longitudes):
                # Convert lat/lon to row/col
                row, col = src.index(lon, lat)

                # Check if within bounds
                if 0 <= row < src.height and 0 <= col < src.width:
                    value = data[row, col]
                else:
                    value = np.nan

                sampled_values.append(value)

        return np.array(sampled_values)

    except Exception as e:
        print(f"Error sampling raster: {str(e)}")
        return np.zeros(len(latitudes))


def load_srtm_elevation(
    df: pd.DataFrame,
    srtm_path: str,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> pd.DataFrame:
    """
    Load SRTM DEM elevation data and sample at coordinates.

    Args:
        df: DataFrame with coordinates
        srtm_path: Path to SRTM DEM GeoTIFF
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        DataFrame with added elevation column
    """
    df = df.copy()

    print(f"Sampling elevation from {srtm_path}...")

    elevations = sample_raster_at_points(
        srtm_path,
        df[lat_col].values,
        df[lon_col].values
    )

    df['elevation'] = elevations
    print(f"Elevation stats: min={df['elevation'].min():.1f}, max={df['elevation'].max():.1f}, mean={df['elevation'].mean():.1f}")

    return df


def calculate_slope(elevation_array: np.ndarray, resolution: float = 30.0) -> np.ndarray:
    """
    Calculate slope from elevation data using Sobel filter.

    Args:
        elevation_array: 2D array of elevation values
        resolution: Spatial resolution in meters

    Returns:
        Array of slope values in degrees
    """
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not available. Returning zeros for slope.")
        return np.zeros_like(elevation_array)

    # Calculate gradients
    dx = sobel(elevation_array, axis=1) / (8 * resolution)
    dy = sobel(elevation_array, axis=0) / (8 * resolution)

    # Calculate slope in degrees
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi

    return slope


def compute_terrain_features(
    df: pd.DataFrame,
    dem_path: str,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> pd.DataFrame:
    """
    Compute terrain features from DEM: elevation, slope, aspect.

    Args:
        df: DataFrame with coordinates
        dem_path: Path to DEM GeoTIFF
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        DataFrame with added terrain features
    """
    df = df.copy()

    if not RASTERIO_AVAILABLE:
        print("Warning: rasterio not available. Creating dummy terrain features.")
        df['elevation'] = 0
        df['slope'] = 0
        df['aspect'] = 0
        return df

    print("Computing terrain features from DEM...")

    try:
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)

            # Sample elevation
            elevations = sample_raster_at_points(
                dem_path,
                df[lat_col].values,
                df[lon_col].values
            )
            df['elevation'] = elevations

            # Calculate slope and aspect for the entire DEM
            if SCIPY_AVAILABLE:
                slope_data = calculate_slope(dem_data)

                # Sample slope at points
                slopes = []
                for lat, lon in zip(df[lat_col], df[lon_col]):
                    row, col = src.index(lon, lat)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        slopes.append(slope_data[row, col])
                    else:
                        slopes.append(0)

                df['slope'] = slopes
            else:
                df['slope'] = 0

            # Aspect (simplified)
            df['aspect'] = 0

    except Exception as e:
        print(f"Error computing terrain features: {str(e)}")
        df['elevation'] = 0
        df['slope'] = 0
        df['aspect'] = 0

    print("Terrain features added")
    return df


def load_esa_worldcover(
    df: pd.DataFrame,
    worldcover_path: str,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> pd.DataFrame:
    """
    Load ESA WorldCover land cover data and sample at coordinates.

    Args:
        df: DataFrame with coordinates
        worldcover_path: Path to ESA WorldCover GeoTIFF
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        DataFrame with added land_cover column
    """
    df = df.copy()

    print(f"Sampling land cover from {worldcover_path}...")

    land_cover = sample_raster_at_points(
        worldcover_path,
        df[lat_col].values,
        df[lon_col].values
    )

    df['land_cover'] = land_cover.astype(int)

    # ESA WorldCover classes
    landcover_classes = {
        10: 'Tree cover',
        20: 'Shrubland',
        30: 'Grassland',
        40: 'Cropland',
        50: 'Built-up',
        60: 'Bare/sparse vegetation',
        70: 'Snow and ice',
        80: 'Permanent water bodies',
        90: 'Herbaceous wetland',
        95: 'Mangroves',
        100: 'Moss and lichen'
    }

    print("Land cover distribution:")
    print(df['land_cover'].value_counts().sort_index())

    return df


def create_distance_features(
    df: pd.DataFrame,
    reference_points: Optional[List[Tuple[float, float]]] = None,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> pd.DataFrame:
    """
    Create distance features to reference points (e.g., cities, water bodies).

    Args:
        df: DataFrame with coordinates
        reference_points: List of (lat, lon) tuples for reference locations
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        DataFrame with added distance features
    """
    df = df.copy()

    if reference_points is None:
        # Example reference points (replace with actual data)
        reference_points = [
            (40.7128, -74.0060),  # New York
            (34.0522, -118.2437),  # Los Angeles
            (41.8781, -87.6298),   # Chicago
        ]

    for i, (ref_lat, ref_lon) in enumerate(reference_points):
        # Haversine distance approximation
        df[f'dist_to_ref_{i}'] = haversine_distance(
            df[lat_col].values,
            df[lon_col].values,
            ref_lat,
            ref_lon
        )

    print(f"Created distance features to {len(reference_points)} reference points")
    return df


def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float
) -> np.ndarray:
    """
    Calculate Haversine distance between points.

    Args:
        lat1: Array of latitudes
        lon1: Array of longitudes
        lat2: Reference latitude
        lon2: Reference longitude

    Returns:
        Array of distances in kilometers
    """
    R = 6371  # Earth's radius in kilometers

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    distance = R * c
    return distance


def create_spatial_clusters(
    df: pd.DataFrame,
    n_clusters: int = 10,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> pd.DataFrame:
    """
    Create spatial clusters using K-means.

    Args:
        df: DataFrame with coordinates
        n_clusters: Number of clusters
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        DataFrame with added spatial_cluster column
    """
    df = df.copy()

    try:
        from sklearn.cluster import KMeans

        coords = df[[lat_col, lon_col]].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['spatial_cluster'] = kmeans.fit_predict(coords)

        print(f"Created {n_clusters} spatial clusters")
    except ImportError:
        print("Warning: sklearn not available. Skipping spatial clustering.")
        df['spatial_cluster'] = 0

    return df


def create_watershed_features(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> pd.DataFrame:
    """
    Create watershed-based features (simplified grid-based approach).

    Args:
        df: DataFrame with coordinates
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        DataFrame with added watershed features
    """
    df = df.copy()

    # Simple grid-based watershed assignment
    lat_bins = pd.cut(df[lat_col], bins=20, labels=False)
    lon_bins = pd.cut(df[lon_col], bins=20, labels=False)

    df['watershed_id'] = lat_bins * 20 + lon_bins

    print("Created watershed features")
    return df
