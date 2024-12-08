import socket

# Configuration
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 11123      # Replace with the port used by the GPS2IP app

def parse_gprmc(nmea_sentence):
    try:
        # Split the NMEA sentence into fields
        fields = nmea_sentence.split(',')

        # Check the sentence type
        if fields[0] != '$GPRMC':
            print("Not a GPRMC sentence")
            return None

        # Extract latitude and longitude fields
        raw_latitude = fields[3]
        latitude_direction = fields[4]
        raw_longitude = fields[5]
        longitude_direction = fields[6]

        # Convert latitude from DDMM.MMMM to decimal degrees
        lat_degrees = int(raw_latitude[:2])  # First two digits are degrees
        lat_minutes = float(raw_latitude[2:])  # Remaining are minutes
        latitude = lat_degrees + (lat_minutes / 60)
        if latitude_direction == 'S':
            latitude = -latitude  # South is negative

        # Convert longitude from DDDMM.MMMM to decimal degrees
        lon_degrees = int(raw_longitude[:3])  # First three digits are degrees
        lon_minutes = float(raw_longitude[3:])  # Remaining are minutes
        longitude = lon_degrees + (lon_minutes / 60)
        if longitude_direction == 'W':
            longitude = -longitude  # West is negative

        return latitude, longitude

    except (IndexError, ValueError) as e:
        print(f"Error parsing NMEA sentence: {e}")
        return None, None

def receive_gps_data():
    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        # Bind to the specified host and port
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Listening for connections on {HOST}:{PORT}...")

        # Accept a client connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")

        with client_socket:
            # Receive data from the client
            data = client_socket.recv(1024)  # Adjust buffer size if necessary
            if not data:
                print("Failed to reiceive data from phone")
                # Decode ASCII message
                return None, None
            else:
                gps_data = data.decode('ascii').strip()
                latitude, longitude = parse_gprmc(gps_data)
                if latitude is not None and longitude is not None:
                    print(f"Received GPS data: {latitude}, {longitude}")
                    return latitude, longitude
                else:
                    print(f"Failed to receive latitude and longitude")
                    return None, None



if __name__ == "__main__":
    receive_gps_data()