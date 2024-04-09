from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
import pandas as pd
import os
import time

os.environ['PIP_DEFAULT_TIMEOUT'] = '1000'
geolocator = Nominatim(user_agent="Delany_Test")

def do_geocode(address, times, max_times = 5):
    if times > max_times:
        return("NA")
    try:
        location = geolocator.geocode(address)
        if location == None:
            return "NA"
        return (location.latitude, location.longitude)
    except GeocoderUnavailable:
        time.sleep(2)
        print("Failed")
        do_geocode(address, times + 1,  max_times)

test_data= pd.read_csv("C:/Users/William/Downloads/test_data.csv", sep = ',')
for address in test_data["Incident Address"]:
    address = address + ", Iowa City, IA"
    print(do_geocode(address, 0, 1))
    time.sleep(2)


