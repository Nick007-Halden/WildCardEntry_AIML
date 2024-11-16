from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import folium

def get_coordinates(location_name):
    """Get coordinates (latitude, longitude) for a given location name."""
    try:
        geolocator = Nominatim(user_agent="geo_distance_calculator")
        location = geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"Could not find the location: {location_name}. Please try again.")
            return None
    except Exception as e:
        print(f"Error occurred while fetching coordinates: {e}")
        return None

def main():
    # Input source and destination
    source = input("Enter the source location: ")
    destination = input("Enter the destination location: ")

    # Get coordinates for source and destination
    source_coords = get_coordinates(source)
    destination_coords = get_coordinates(destination)

    if not source_coords or not destination_coords:
        print("Failed to fetch coordinates. Exiting...")
        return

    # Calculate distance
    distance = geodesic(source_coords, destination_coords).kilometers
    print(f"The distance between {source} and {destination} is {distance:.2f} kilometers.")

    # Create map centered at the midpoint between source and destination
    midpoint = [(source_coords[0] + destination_coords[0]) / 2,
                (source_coords[1] + destination_coords[1]) / 2]
    map_object = folium.Map(location=midpoint, zoom_start=6)

    # Add markers for source and destination
    folium.Marker(source_coords, popup=f"Source: {source}").add_to(map_object)
    folium.Marker(destination_coords, popup=f"Destination: {destination}").add_to(map_object)

    # Add a line between source and destination
    folium.PolyLine([source_coords, destination_coords], color="blue", weight=2.5, opacity=1).add_to(map_object)

    # Add distance label
    folium.Marker(
        midpoint,
        popup=f"Distance: {distance:.2f} km",
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black;">{distance:.2f} km</div>')
    ).add_to(map_object)

    # Save the map to an HTML file
    map_file = "distance_map.html"
    map_object.save(map_file)
    print(f"Map has been saved as {map_file}. Open it in your browser to view.")

if __name__ == "__main__":
    main()
