from flask_sqlalchemy import SQLAlchemy
from flask import Flask, jsonify
import requests
from datetime import datetime

FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "geo_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "baro_altitude", "squawk",
    "spi", "position_source", "category"
]
updated_at = None
staleness_threshold = 10 # Minutes

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flights.db'
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

# Get all flights
@app.route('/flights')
def get_flights():
    update_flights()
    flight_objs = Flight.query.all()
    flight_arr = [f.to_dict() for f in flight_objs]
    return jsonify(flight_arr)

# Get a flight 
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


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        update_flights()
    app.run(debug=True)