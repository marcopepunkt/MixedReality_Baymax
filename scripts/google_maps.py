import googlemaps
from datetime import datetime
import re
import receive_gps
from math import radians, sin, cos, sqrt, atan2

# TODO: insert key of your Google Cloud/Maps platform (need to make an account)
class GoogleMapsClient:
    def __init__(self,args_api_key):
        self.gmaps = googlemaps.Client(key=args_api_key)


    def time_difference_in_hours_and_minutes(self,time1, time2):
        """
        used for calculating how much time is left before a tram departs.
        Args:
            time1: given as e.g. 13:15PM
            time2: given as e.g. 13:15PM

        Returns: hours, minutes

        """
        # Define the time format
        time_format = "%I:%M %p"

        # Parse the time strings into datetime objects
        t1 = datetime.strptime(time1, time_format)
        t2 = datetime.strptime(time2, time_format)

        # Calculate the difference
        difference = t2 - t1

        # Get total seconds and convert to hours and minutes
        total_seconds = difference.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)

        return hours, minutes


    def haversine(self,lat1, lon1, lat2, lon2):
        """
        Used for comparing user's current gps coordinates to a target gps coordinates.
        Calculate the great-circle (shortest) distance in meters between two GPS coordinates.

        Parameters:
        lat1, lon1: Latitude and Longitude of the first point (in decimal degrees)
        lat2, lon2: Latitude and Longitude of the second point (in decimal degrees)

        Returns:
        Distance in meters.
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        radius_of_earth = 6371000  # Earth's radius in meters
        distance = radius_of_earth * c
        return distance # in meters


    def get_main_directions(self,destination):
        """
        Used for getting main directions ("Walk to tram stop Unispital, take tram 10 at 13:00, ...") to a destination.

        Args:
            destination: address, given as a string of words

        Returns:
            instructions: main directions as above
            stop_coordinates: list containing objects of shape
                            {"name": departure_stop, "gps_coords": f"{departure_stop_lat},{departure_stop_lng}", "departure_time": departure_time}
                            for each tram stop and final destination

        """
        origin_latitude, origin_longitude = receive_gps.receive_gps_data()
        origin = f"{origin_latitude},{origin_longitude}"
        print("destination is " + destination)
        directions_result = self.gmaps.directions(origin,
                                            destination + ",Zuerich",
                                            mode="transit",
                                            transit_mode="rail")

        if directions_result == []:
            print("Failed to compute directions to destination address")
            return None, None

        stop_coordinates = []
        instructions = ""

        for index_0 in range(len(directions_result[0]["legs"][0]["steps"])):
            step_0 = directions_result[0]["legs"][0]["steps"][index_0]
            main_instruction = step_0["html_instructions"]
            main_cleaned_instruction = re.sub(r'<.*?>', ' ', main_instruction)
            main_cleaned_instruction = main_cleaned_instruction + "."
            if index_0 > 0 and index_0 != len(directions_result[0]["legs"][0]["steps"]) - 1:
                main_cleaned_instruction = "Then, " + main_cleaned_instruction

            # take tram/bus/train instruction. print the stops' names, time too
            if any(keyword in main_cleaned_instruction.lower() for keyword in ["tram", "train", "bus"]):
                vehicle_name = step_0["transit_details"]["line"]["vehicle"]["name"]
                line_name = step_0["transit_details"]["line"]["short_name"]
                arrival_stop = step_0["transit_details"]["arrival_stop"]["name"]
                departure_stop = step_0["transit_details"]["departure_stop"]["name"]
                departure_time = step_0["transit_details"]["departure_time"]["text"]
                current_time = datetime.now()
                current_time = current_time.strftime("%I:%M %p")
                hours, minutes = self.time_difference_in_hours_and_minutes(current_time, departure_time)
                departure_stop_lat = step_0["transit_details"]["departure_stop"]["location"]["lat"]
                departure_stop_lng = step_0["transit_details"]["departure_stop"]["location"]["lng"]
                stop_coordinates.append({'name': departure_stop, 'gps_coords': f"{departure_stop_lat},{departure_stop_lng}", 'departure_time': departure_time})
                if hours == 0:
                    duration = f"in {minutes} minutes."
                else:
                    duration = f"in {hours} hours and {minutes} minutes."
                main_cleaned_instruction = f"Then, take {vehicle_name} {line_name} from {departure_stop} to {arrival_stop} at {departure_time}, so " + duration

            else:
                # if it's last instruction, remove the "Zurich, Switzerland", just print destination name
                if index_0 == len(directions_result[0]["legs"][0]["steps"]) - 1:
                    parts = main_cleaned_instruction.split(',')
                    if len(parts) >= 3:
                        main_cleaned_instruction = ','.join(parts[:len(parts) - 2])
                    main_cleaned_instruction = "Lastly, " + main_cleaned_instruction + "."

            instructions = instructions + main_cleaned_instruction

        destination_lat = directions_result[0]["legs"][0]["end_location"]["lat"]
        destination_long = directions_result[0]["legs"][0]["end_location"]["lng"]
        stop_coordinates.append({'name': destination, 'gps_coords': f"{destination_lat},{destination_long}", 'departure_time': ""})

        time_of_arrival = directions_result[0]["legs"][0]["arrival_time"]["text"]
        current_time = datetime.now()
        current_time = current_time.strftime("%I:%M %p")
        hours, minutes = self.time_difference_in_hours_and_minutes(current_time, time_of_arrival)
        if hours == 0:
            duration = f"in {str(minutes)} minutes."
        else:
            duration = f"in {str(hours)} hours and {str(minutes)} minutes."
        instructions = instructions + f"You will arrive at {time_of_arrival}, or {duration}"

        print("Returning following instructions:" + instructions)
        print("length of stop coordinates:" + str(len(stop_coordinates)))
        return instructions, stop_coordinates


    def get_walking_directions(self,destination_coords):
        """
        returns walking directions to tram stop or final destination (e.g. "Walk straight for 10 meters. Then turn right...")
        Args:
            destination_coords: a string of gps coordinates in the form of "{latitude},{longitude}"

        Returns:
            subinstructions: list of objects of shape
                            {"instruction": instruction, "gps_lat": lat, "gps_lng": lng, "distance": distance}
                            for each intermediate waypoint on the way to a tram stop

        """
        origin_latitude, origin_longitude = receive_gps.receive_gps_data()
        origin = f"{origin_latitude},{origin_longitude}"
        directions_first_stop = self.gmaps.directions(origin,
                                            destination_coords,
                                            mode="walking",
                                            avoid="indoor")
        if len(directions_first_stop) == 0 or directions_first_stop is None:
            print("Could not get directions")
            return None

        subinstructions = []

        for index_0 in range(len(directions_first_stop[0]["legs"][0]["steps"])):
            step_0 = directions_first_stop[0]["legs"][0]["steps"][index_0]
            main_instruction = step_0["html_instructions"]
            main_cleaned_instruction = re.sub(r'<.*?>', ' ', main_instruction)
            distance = step_0["distance"]["text"]
            duration = step_0["duration"]["text"]
            instruction = main_cleaned_instruction + f"in {duration}, or {distance}."
            print(instruction)
            lat = step_0["end_location"]["lat"]
            lng = step_0["end_location"]["lng"]
            subinstructions.append({"instruction": instruction, "gps_lat": lat, "gps_lng": lng, "distance": distance})

        return subinstructions

    def compute_distance_to_target(self,target_lat, target_lng):
        target_lat = float(target_lat)
        target_lng = float(target_lng)
        origin_latitude, origin_longitude = receive_gps.receive_gps_data()
        origin_latitude = float(origin_latitude)
        origin_longitude = float(origin_longitude)

        distance = self.haversine(target_lat, target_lng, origin_latitude, origin_longitude)
        return distance




