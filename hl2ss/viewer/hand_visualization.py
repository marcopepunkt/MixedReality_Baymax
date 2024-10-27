import numpy as np
from pynput import keyboard
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import http.server
import socketserver
import threading
import json
import webbrowser
from datetime import datetime
import os
from get_right_hand_pose import normalize_vector, calculate_finger_angles, get_finger_joint_positions

# Keep existing helper functions and hand tracking code...
# (normalize_vector, calculate_finger_angles, get_finger_joint_positions)

class HandTrackingVisualizer:
    def __init__(self, hl2_host="169.254.174.24", http_port=8000):
        self.hl2_host = hl2_host
        self.http_port = http_port
        self.enable = True
        self.current_hand_data = None
        
        # Create HTML file with visualization
        self.html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Hand Tracking Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status {
            padding: 5px 10px;
            border-radius: 4px;
            display: inline-block;
            margin-bottom: 10px;
        }
        .connected {
            background-color: #d4edda;
            color: #155724;
        }
        .disconnected {
            background-color: #f8d7da;
            color: #721c24;
        }
        .timestamp {
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .finger-data {
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Hand Tracking Visualization</h1>
        <div id="status" class="status disconnected">Disconnected</div>
        <div id="visualization">
            <svg width="400" height="400" viewBox="0 0 400 400" id="handSvg">
                <!-- Palm -->
                <rect x="150" y="180" width="100" height="120" rx="10" 
                      fill="#e0e0e0" stroke="#888" stroke-width="2"/>
                <!-- Fingers will be drawn here -->
            </svg>
        </div>
        <div id="angles" class="finger-data"></div>
        <div id="timestamp" class="timestamp"></div>
    </div>

    <script>
        const colors = {
            thumb: 'hsl(0, 70%, 50%)',
            index: 'hsl(50, 70%, 50%)',
            middle: 'hsl(100, 70%, 50%)',
            ring: 'hsl(150, 70%, 50%)',
            little: 'hsl(200, 70%, 50%)'
        };

        const fingerStartPositions = {
            thumb: [160, 220],
            index: [180, 180],
            middle: [200, 180],
            ring: [220, 180],
            little: [240, 180]
        };

        function calculateFingerPath(startX, startY, angles) {
            const SEGMENT_LENGTH = 30;
            let points = [[startX, startY]];
            let currentAngle = 0;
            
            // MCP joint
            currentAngle = angles.MCP_flexion;
            const mcpX = startX + Math.sin(currentAngle * Math.PI / 180) * SEGMENT_LENGTH;
            const mcpY = startY - Math.cos(currentAngle * Math.PI / 180) * SEGMENT_LENGTH;
            points.push([mcpX, mcpY]);
            
            // PIP joint
            currentAngle += angles.PIP;
            const pipX = mcpX + Math.sin(currentAngle * Math.PI / 180) * SEGMENT_LENGTH;
            const pipY = mcpY - Math.cos(currentAngle * Math.PI / 180) * SEGMENT_LENGTH;
            points.push([pipX, pipY]);
            
            // DIP joint
            currentAngle += angles.DIP;
            const dipX = pipX + Math.sin(currentAngle * Math.PI / 180) * SEGMENT_LENGTH;
            const dipY = pipY - Math.cos(currentAngle * Math.PI / 180) * SEGMENT_LENGTH;
            points.push([dipX, dipY]);
            
            return points;
        }

        function updateVisualization(handData) {
            const svg = document.getElementById('handSvg');
            
            // Clear previous fingers
            const existingFingers = svg.querySelectorAll('.finger');
            existingFingers.forEach(finger => finger.remove());
            
            // Draw each finger
            Object.entries(handData).forEach(([fingerName, angles]) => {
                const [startX, startY] = fingerStartPositions[fingerName];
                const points = calculateFingerPath(startX, startY, angles);
                
                // Create finger group
                const fingerGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                fingerGroup.classList.add('finger');
                
                // Create path
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('d', `M ${points.map(p => p.join(' ')).join(' L ')}`);
                path.setAttribute('stroke', colors[fingerName]);
                path.setAttribute('stroke-width', '4');
                path.setAttribute('fill', 'none');
                path.setAttribute('stroke-linecap', 'round');
                path.setAttribute('stroke-linejoin', 'round');
                
                // Add joints
                points.forEach(([x, y]) => {
                    const joint = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    joint.setAttribute('cx', x);
                    joint.setAttribute('cy', y);
                    joint.setAttribute('r', '3');
                    joint.setAttribute('fill', colors[fingerName]);
                    fingerGroup.appendChild(joint);
                });
                
                fingerGroup.appendChild(path);
                svg.appendChild(fingerGroup);
            });
            
            // Update angles display
            const anglesDiv = document.getElementById('angles');
            anglesDiv.innerHTML = Object.entries(handData)
                .map(([finger, angles]) => `
                    <div style="color: ${colors[finger]}">
                        ${finger}: 
                        DIP ${angles.DIP.toFixed(1)}° 
                        PIP ${angles.PIP.toFixed(1)}° 
                        MCP ${angles.MCP_flexion.toFixed(1)}°
                        ADD ${angles.ADD.toFixed(1)}°
                    </div>
                `).join('');
        }

        function updateData() {
            fetch('/hand-data')
                .then(response => response.json())
                .then(data => {
                    if (data.handData) {
                        document.getElementById('status').className = 'status connected';
                        document.getElementById('status').textContent = 'Connected';
                        document.getElementById('timestamp').textContent = 
                            `Last updated: ${new Date(data.timestamp).toLocaleTimeString()}`;
                        updateVisualization(data.handData);
                    }
                })
                .catch(error => {
                    document.getElementById('status').className = 'status disconnected';
                    document.getElementById('status').textContent = 'Disconnected';
                    console.error('Error fetching hand data:', error);
                });
        }

        // Update visualization every 16ms (~60fps)
        setInterval(updateData, 16);
    </script>
</body>
</html>
'''
        
        # Write HTML file
        with open('hand_visualization.html', 'w') as f:
            f.write(self.html_content)

    def start_http_server(self):
        """Start HTTP server to serve visualization page and hand data"""
        class HandDataHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/hand-data':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    if self.server.hand_tracking_viz.current_hand_data:
                        data = {
                            'timestamp': datetime.now().isoformat(),
                            'handData': self.server.hand_tracking_viz.current_hand_data
                        }
                        self.wfile.write(json.dumps(data).encode())
                else:
                    return super().do_GET()

        class HandTrackingServer(socketserver.TCPServer):
            def __init__(self, server_address, RequestHandlerClass, hand_tracking_viz):
                self.hand_tracking_viz = hand_tracking_viz
                super().__init__(server_address, RequestHandlerClass)

        server = HandTrackingServer(('', self.http_port), HandDataHandler, self)
        print(f"Starting HTTP server at http://localhost:{self.http_port}")
        
        # Open web browser
        webbrowser.open(f'http://localhost:{self.http_port}/hand_visualization.html')
        
        # Start server in a thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

    def start_hand_tracking(self):
        """Start the hand tracking loop"""
        client = hl2ss_lnm.rx_si(self.hl2_host, hl2ss.StreamPort.SPATIAL_INPUT)
        client.open()

        print("Starting hand tracking. Press ESC to stop.")
        
        def on_press(key):
            if key == keyboard.Key.esc:
                self.enable = False
            return self.enable

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        while self.enable:
            data = client.get_next_packet()
            si = hl2ss.unpack_si(data.payload)

            if si.is_valid_hand_right():
                hand_right = si.get_hand_right()
                palm_position = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Palm).position
                
                # Calculate palm normal
                wrist_position = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Wrist).position
                middle_metacarpal = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.MiddleMetacarpal).position
                palm_vector = middle_metacarpal - wrist_position
                side_vector = (hand_right.get_joint_pose(hl2ss.SI_HandJointKind.LittleMetacarpal).position - 
                             hand_right.get_joint_pose(hl2ss.SI_HandJointKind.ThumbMetacarpal).position)
                palm_normal = normalize_vector(np.cross(palm_vector, side_vector))
                
                # Get angles for each finger
                hand_data = {}
                for finger in ['thumb', 'index', 'middle', 'ring', 'little']:
                    joint_positions = get_finger_joint_positions(hand_right, finger)
                    angles = calculate_finger_angles(joint_positions, palm_position, palm_normal)
                    hand_data[finger] = angles
                
                self.current_hand_data = hand_data

        client.close()
        listener.join()

    def run(self):
        """Run both the HTTP server and hand tracking"""
        self.start_http_server()
        self.start_hand_tracking()

if __name__ == "__main__":
    visualizer = HandTrackingVisualizer()
    visualizer.run()