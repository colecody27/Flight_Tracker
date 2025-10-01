
import os
from flask import Flask, jsonify, request
import requests
from datetime import datetime
import math 
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_, or_

FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "geo_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "baro_altitude", "squawk",
    "spi", "position_source", "category"
]
updated_at = None
staleness_threshold = 10 # Minutes

app = Flask(__name__)
# Allow DATABASE_URL to be set in the environment for deploys (e.g. Postgres).
# Fallback to local sqlite for development.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///flights.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Flight(db.Model):
    __tablename__ = "flights"

    icao24 = db.Column(db.String(6), primary_key=True)
    callsign = db.Column(db.String(10))
    origin_country = db.Column(db.String(50))
    time_position = db.Column(db.BigInteger)
    last_contact = db.Column(db.BigInteger)
    longitude = db.Column(db.Float)
    latitude = db.Column(db.Float)
    geo_altitude = db.Column(db.Float)
    on_ground = db.Column(db.Boolean)
    velocity = db.Column(db.Float)
    true_track = db.Column(db.Float)
    vertical_rate = db.Column(db.Float)
    baro_altitude = db.Column(db.Float)
    squawk = db.Column(db.String(10))
    spi = db.Column(db.Boolean)
    position_source = db.Column(db.Integer)

    def to_dict(self):
        return {
            "icao24": self.icao24,
            "callsign": self.callsign,
            "origin_country": self.origin_country,
            "time_position": self.time_position,
            "last_contact": self.last_contact,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "geo_altitude": self.geo_altitude,
            "on_ground": self.on_ground,
            "velocity": self.velocity,
            "true_track": self.true_track,
            "vertical_rate": self.vertical_rate,
            "baro_altitude": self.baro_altitude,
            "squawk": self.squawk,
            "spi": self.spi,
            "position_source": self.position_source
        }
    
class Distance(db.Model):
    __tablename__ = "distance"

    icao24_1 = db.Column(db.String(6), primary_key=True)
    icao24_2 = db.Column(db.String(6), primary_key=True)
    distance = db.Column(db.Float)
    # closest_time = db.Column(db.DateTime)

    def to_dict(self):
        return {
            "icao24_1": self.icao24_1,
            "icao24_2": self.icao24_2,
            "distance": self.distance,
            # "closest_time": self.closest_time
        }

# Get flights based on query params or default to all
@app.route('/flights/search')
def search_flights():
    # optional query params
    icao24 = request.args.get('icao24')      
    callsign = request.args.get('callsign')        # string, partial match
    airlines = request.args.get('airlines')        # string or comma-separated
    max_alt = request.args.get('max_alt')          # numeric
    max_vel = request.args.get('max_vel')          # numeric
    origins = request.args.get('origins')          # string or comma-separated

    filters = []

    if icao24:
        filters.append(Flight.icao24 == icao24)

    if callsign:
        filters.append(Flight.callsign.ilike(f'%{callsign}%'))
        
    if max_alt:
        try:
            max_alt_val = float(max_alt)
            filters.append(Flight.geo_altitude <= max_alt_val)
        except ValueError:
            return jsonify({'error': 'max_alt must be numeric'}), 400

    if max_vel:
        try:
            max_vel_val = float(max_vel)
            filters.append(Flight.velocity <= max_vel_val)
        except ValueError:
            return jsonify({'error': 'max_vel must be numeric'}), 400

    if origins:
        # allow comma-separated list of origins, ORed together
        origin_list = [o.strip() for o in origins.split(',') if o.strip()]
        if origin_list:
            filters.append(or_(*(Flight.origin_country == o for o in origin_list)))

    if airlines:
        # if airlines maps to callsign prefixes or similar, adapt accordingly.
        # Example: match callsign starting with any of the comma-separated airline codes
        airline_list = [a.strip() for a in airlines.split(',') if a.strip()]
        if airline_list:
            filters.append(or_(*(Flight.callsign.ilike(f'{a}%') for a in airline_list)))

    # If no filters, return everything
    query = Flight.query
    if filters:
        query = query.filter(*filters)

    results = query.all()
    return jsonify([f.to_dict() for f in results])

# Get distances 
@app.route('/flights/<id>')
def get_flight(id):
    update_flights()
    flight_obj = Flight.query.get(id)
    return (jsonify(flight_obj.to_dict())) if flight_obj else (jsonify({'error': f'icao24: {id} not found'}), 404)

# Update DB if data is older than 10 minutes
def update_flights():
    global updated_at
    staleness_in_minutes = (datetime.now() - updated_at).total_seconds()/60 if updated_at else None
    if not staleness_in_minutes or staleness_in_minutes > staleness_threshold:
        response = requests.get('https://opensky-network.org/api/states/all').json()
        states = response['states']
        
        for state in states:
            flight_data = dict(zip(FIELDS, state))
            del flight_data['sensors']

            flight = Flight.query.get(flight_data['icao24'])
            if flight:
                for key, value in flight_data.items():
                    setattr(flight, key, value)
            else:
                db.session.add(Flight(**flight_data))
        
        updated_at = datetime.now()
        db.session.commit()
        calculate_distances()

@app.route('/flights/distances')
def get_distances():
    max_dist = request.args.get('max')
    if max_dist:
        try:
            distances = Distance.query.filter(Distance.distance <= float(max_dist))
        except ValueError:
            return jsonify({'error': 'Error reading dist_threshold, must be float'})
    else:
        distances = Distance.query.all()
    return jsonify([d.to_dict() for d in distances])


def calculate_distances():
    all_flights = Flight.query.filter(
        Flight.icao24.isnot(None),
        Flight.latitude.isnot(None),
        Flight.longitude.isnot(None),
        Flight.geo_altitude.isnot(None)
    ).all()

    for i, flight_1 in enumerate(all_flights):
        p1 = geodetic_to_ecef(flight_1.latitude, flight_1.longitude, flight_1.geo_altitude)
        # Query for flights that are within a relative range
        close_flights = [f for f in all_flights[i+1:]
                          if abs(flight_1.longitude - f.longitude) < .05
                            and abs(flight_1.latitude - f.latitude) < .05
                            and flight_1.icao24 != f.icao24]
        
        # Calculate distances between them 
        for flight_2 in close_flights:
            p2 = geodetic_to_ecef(flight_2.latitude, flight_2.longitude, flight_2.geo_altitude)
            distance = np.linalg.norm(p2 - p1)
            
            a,b = sorted([flight_1.icao24, flight_2.icao24])
            curr = Distance.query.get((a,b))
            if curr:
                curr.distance = distance
            else:
                db.session.add(Distance(icao24_1=a, icao24_2=b, distance=distance))
    db.session.commit()
        

def geodetic_to_ecef(lat, lon, alt):
    # WGS84 constants
    a = 6378137.0          # semi-major axis (meters)
    f = 1 / 298.257223563  # flattening
    e2 = f * (2 - f)       # eccentricity squared

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)

    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)

    return np.array([x, y, z])

if __name__ == '__main__':
    # Create DB tables on startup; don't let a failed update_flights() abort the server
    with app.app_context():
        db.create_all()
        try:
            update_flights()
        except Exception as e:
            # Log and continue - in many hosting environments the OpenSky API
            # may be rate-limited or blocked during deploys.
            print(f"Warning: initial update_flights() failed: {e}")

    # Bind to 0.0.0.0 so container/platforms can reach the server. Allow PORT override.
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug_flag = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host=host, port=port, debug=debug_flag)