import googlemaps
from datetime import datetime
import requests
import html
import re
import receive_gps

# TODO: insert key of your Google Cloud/Maps platform (need to make an account)
KEY = 'insert your key'
gmaps = googlemaps.Client(key=KEY)

# directions_first_stop = gmaps.directions(origin,
#                                      stop_coordinates[0],
#                                      mode="walking",
#                                      avoid="indoor")
#
#
# for index_0 in range(len(directions_first_stop[0]["legs"][0]["steps"])):
#     step_0 = directions_first_stop[0]["legs"][0]["steps"][index_0]
#     main_instruction = step_0["html_instructions"]
#     main_cleaned_instruction = re.sub(r'<.*?>', ' ', main_instruction)
#     distance = step_0["distance"]["text"]
#     duration = step_0["duration"]["text"]
#     print(main_cleaned_instruction + f"in {duration}, or {distance}.")

def time_difference_in_hours_and_minutes(time1, time2):
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


def get_main_directions(destination):
    origin_latitude, origin_longitude = receive_gps.receive_gps_data()
    origin = f"{origin_latitude},{origin_longitude}"
    print("destination is " + destination)
    directions_result = gmaps.directions(origin,
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
        if index_0 > 0 and index_0 != len(directions_result[0]["legs"][0]["steps"]) - 1:
            main_cleaned_instruction = "Then, " + main_cleaned_instruction + "."

        # take tram/bus/train instruction. print the stops' names, time too
        if any(keyword in main_cleaned_instruction.lower() for keyword in ["tram", "train", "bus"]):
            vehicle_name = step_0["transit_details"]["line"]["vehicle"]["name"]
            line_name = step_0["transit_details"]["line"]["short_name"]
            arrival_stop = step_0["transit_details"]["arrival_stop"]["name"]
            departure_stop = step_0["transit_details"]["departure_stop"]["name"]
            departure_time = step_0["transit_details"]["departure_time"]["text"]
            current_time = datetime.now()
            current_time = current_time.strftime("%I:%M %p")
            hours, minutes = time_difference_in_hours_and_minutes(current_time, departure_time)
            departure_stop_lat = step_0["transit_details"]["departure_stop"]["location"]["lat"]
            departure_stop_lng = step_0["transit_details"]["departure_stop"]["location"]["lng"]
            print("name of stop:" + departure_stop)
            print("lat of stop:" + str(departure_stop_lat))
            print("lng of stop:" + str(departure_stop_lng))
            stop_coordinates.append({"name": departure_stop, "gps_coords": f"{departure_stop_lat},{departure_stop_lng}"})
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
        print(main_cleaned_instruction)

    destination_lat = directions_result[0]["legs"][0]["end_location"]["lat"]
    destination_long = directions_result[0]["legs"][0]["end_location"]["lng"]
    stop_coordinates.append({"name": destination, "gps_coords": f"{destination_lat},{destination_long}"})
    print(stop_coordinates[-1])

    time_of_arrival = directions_result[0]["legs"][0]["arrival_time"]["text"]
    current_time = datetime.now()
    current_time = current_time.strftime("%I:%M %p")
    hours, minutes = time_difference_in_hours_and_minutes(current_time, time_of_arrival)
    if hours == 0:
        duration = f"in {str(minutes)} minutes."
    else:
        duration = f"in {str(hours)} hours and {str(minutes)} minutes."
    instructions = instructions + f"You will arrive at {time_of_arrival}, or {duration}"

    print("Returning following instructions:" + instructions)
    print(len(stop_coordinates))
    return instructions, stop_coordinates

