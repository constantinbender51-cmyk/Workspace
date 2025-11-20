import requests
import os

def fetch_weather_salzburg():
    """
    Fetch current weather data for Salzburg
    
    Returns:
        dict: Weather data for Salzburg
    """
    # Get API key from environment variable
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Error: OPENWEATHER_API_KEY environment variable not set")
        print("Please set your OpenWeatherMap API key as an environment variable")
        return None
    
    city = "Salzburg"
    country_code = "AT"
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': f"{city},{country_code}",
        'appid': api_key,
        'units': 'metric'  # For Celsius temperature
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if we got valid data
        if data.get('cod') != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
        
        # Extract relevant weather information
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    """Display weather information in a readable format"""
    if weather_data:
        print(f"\n🌤️  Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"   Temperature: {weather_data['temperature']}°C")
        print(f"   Feels like: {weather_data['feels_like']}°C")
        print(f"   Description: {weather_data['description'].title()}")
        print(f"   Humidity: {weather_data['humidity']}%")
        print(f"   Pressure: {weather_data['pressure']} hPa")
        print(f"   Wind Speed: {weather_data['wind_speed']} m/s")
    else:
        print("No weather data available")

if __name__ == "__main__":
    print("Fetching current weather in Salzburg...")
    
    weather_data = fetch_weather_salzburg()
    
    if weather_data:
        display_weather(weather_data)
        print("\nWeather data fetched successfully!")
    else:
        print("Failed to fetch weather data. Please check your API key and internet connection.")