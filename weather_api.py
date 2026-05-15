import requests

API_KEY = "af8355d44eadc8dc6bb68a484e0f822a"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    if "main" not in data or "weather" not in data or "wind" not in data:
        message = data.get("message", "Weather lookup failed.")
        raise ValueError(message)

    weather = {
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "wind": data["wind"]["speed"],
        "description": data["weather"][0]["description"],
        "icon": data["weather"][0]["icon"]
    }

    return weather
