import atexit
import json
import os
import re
import smtplib
import sqlite3
from collections import defaultdict
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import tensorflow as tf
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, flash, redirect, render_template, request, session, url_for
from PIL import Image

from database import FARMERS_DB, PLANNER_DB, create_db
from weather_api import get_weather

try:
    from twilio.rest import Client
except ImportError:
    Client = None

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "farmai-secret-key")

create_db()

# Load trained model
model = tf.keras.models.load_model("model/crop_model.h5")

# Load class names
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

# Load advisory dataset
with open("model/disease_advisory.json", "r") as f:
    advisory = json.load(f)

scheduler = BackgroundScheduler(daemon=True)

LANGUAGES = {
    "en": "English",
    "hi": "हिंदी",
    "mr": "मराठी",
}

TRANSLATIONS = {
    "app_title": {
        "en": "FarmAI Dashboard",
        "hi": "फार्मएआई डैशबोर्ड",
        "mr": "फार्मएआय डॅशबोर्ड",
    },
    "nav_dashboard": {"en": "Dashboard", "hi": "डैशबोर्ड", "mr": "डॅशबोर्ड"},
    "nav_scan": {"en": "Scan Crop", "hi": "फसल स्कैन", "mr": "पीक स्कॅन"},
    "nav_weather": {"en": "Weather", "hi": "मौसम", "mr": "हवामान"},
    "nav_planner": {"en": "Planner", "hi": "प्लानर", "mr": "नियोजक"},
    "nav_crops": {"en": "My Crops", "hi": "मेरी फसलें", "mr": "माझी पिके"},
    "nav_insights": {"en": "AI Insights", "hi": "एआई इनसाइट्स", "mr": "एआय इनसाइट्स"},
    "nav_login": {"en": "Farmer Login", "hi": "किसान लॉगिन", "mr": "शेतकरी लॉगिन"},
    "nav_signup": {"en": "Create Account", "hi": "खाता बनाएं", "mr": "खाते तयार करा"},
    "language_label": {"en": "Language", "hi": "भाषा", "mr": "भाषा"},
    "dashboard_title": {"en": "Dashboard", "hi": "डैशबोर्ड", "mr": "डॅशबोर्ड"},
    "overview_label": {"en": "FarmAI Overview", "hi": "फार्मएआई अवलोकन", "mr": "फार्मएआय आढावा"},
    "hello_farmer": {"en": "Hello Farmer", "hi": "नमस्ते किसान", "mr": "नमस्कार शेतकरी"},
    "dashboard_subtext": {
        "en": "AI-powered crop monitoring, weather awareness, and planning tools in one place.",
        "hi": "एआई आधारित फसल निगरानी, मौसम जानकारी और योजना उपकरण एक ही जगह पर।",
        "mr": "एआय-आधारित पीक निरीक्षण, हवामान माहिती आणि नियोजन साधने एकाच ठिकाणी.",
    },
    "open_planner": {"en": "Open Planner", "hi": "प्लानर खोलें", "mr": "नियोजक उघडा"},
    "scan_ai_title": {"en": "Scan Crop with AI", "hi": "एआई से फसल स्कैन करें", "mr": "एआयने पीक स्कॅन करा"},
    "scan_ai_subtext": {
        "en": "Upload a plant image to detect diseases instantly.",
        "hi": "रोगों का तुरंत पता लगाने के लिए पौधे की तस्वीर अपलोड करें।",
        "mr": "रोग पटकन ओळखण्यासाठी वनस्पतीचा फोटो अपलोड करा.",
    },
    "scan_plant": {"en": "Scan Plant", "hi": "पौधा स्कैन करें", "mr": "वनस्पती स्कॅन करा"},
    "weather_card_title": {"en": "Weather", "hi": "मौसम", "mr": "हवामान"},
    "my_crops_title": {"en": "My Crops", "hi": "मेरी फसलें", "mr": "माझी पिके"},
    "humidity_label": {"en": "Humidity", "hi": "नमी", "mr": "आर्द्रता"},
    "condition_label": {"en": "Condition", "hi": "स्थिति", "mr": "स्थिती"},
    "ai_insights_title": {"en": "AI Farming Insights", "hi": "एआई खेती इनसाइट्स", "mr": "एआय शेती इनसाइट्स"},
    "day_signals": {"en": "Important crop signals for the day.", "hi": "आज के लिए महत्वपूर्ण फसल संकेत।", "mr": "आजच्या दिवसासाठी महत्त्वाचे पीक संकेत."},
    "high_humidity_detected": {"en": "High humidity detected.", "hi": "उच्च नमी पाई गई।", "mr": "जास्त आर्द्रता आढळली."},
    "fungal_risk_increasing": {"en": "Risk of fungal diseases increasing.", "hi": "फफूंद रोगों का खतरा बढ़ रहा है।", "mr": "बुरशीजन्य रोगांचा धोका वाढत आहे."},
    "monitor_tomato_closely": {"en": "Monitor tomato plants closely.", "hi": "टमाटर के पौधों पर नज़र रखें।", "mr": "टोमॅटो पिकावर बारकाईने लक्ष ठेवा."},
    "scan_page_title": {"en": "Scan Crop", "hi": "फसल स्कैन", "mr": "पीक स्कॅन"},
    "upload_crop_image": {"en": "Upload Crop Image", "hi": "फसल की तस्वीर अपलोड करें", "mr": "पिकाचा फोटो अपलोड करा"},
    "scan_page_subtext": {
        "en": "Upload a crop image to detect plant diseases instantly using AI analysis.",
        "hi": "एआई विश्लेषण से पौधों के रोग तुरंत पहचानने के लिए फसल की तस्वीर अपलोड करें।",
        "mr": "एआय विश्लेषण वापरून वनस्पती रोग पटकन ओळखण्यासाठी पिकाचा फोटो अपलोड करा.",
    },
    "choose_leaf_image": {"en": "Choose a leaf image", "hi": "पत्ते की तस्वीर चुनें", "mr": "पानाचा फोटो निवडा"},
    "file_support": {"en": "PNG, JPG, or JPEG image supported", "hi": "PNG, JPG या JPEG तस्वीर समर्थित है", "mr": "PNG, JPG किंवा JPEG फोटो समर्थित आहे"},
    "scan_crop_button": {"en": "Scan Crop", "hi": "फसल स्कैन करें", "mr": "पीक स्कॅन करा"},
    "detection_result": {"en": "AI Detection Result", "hi": "एआई पहचान परिणाम", "mr": "एआय ओळख परिणाम"},
    "detection_guidance": {"en": "Disease analysis and treatment guidance.", "hi": "रोग विश्लेषण और उपचार मार्गदर्शन।", "mr": "रोग विश्लेषण आणि उपचार मार्गदर्शन."},
    "disease_label": {"en": "Disease", "hi": "रोग", "mr": "रोग"},
    "confidence_label": {"en": "Confidence", "hi": "विश्वास स्तर", "mr": "विश्वास पातळी"},
    "cause_label": {"en": "Cause", "hi": "कारण", "mr": "कारण"},
    "treatment_label": {"en": "Treatment", "hi": "उपचार", "mr": "उपचार"},
    "recommended_pesticides": {"en": "Recommended Pesticides", "hi": "सुझाए गए कीटनाशक", "mr": "शिफारस केलेली कीटकनाशके"},
    "weather_page_title": {"en": "Weather", "hi": "मौसम", "mr": "हवामान"},
    "weather_detection": {"en": "Weather Detection", "hi": "मौसम जानकारी", "mr": "हवामान माहिती"},
    "weather_page_subtext": {
        "en": "Get real-time weather conditions and simple crop guidance for your farm location.",
        "hi": "अपने खेत के स्थान के लिए रियल-टाइम मौसम और सरल फसल मार्गदर्शन प्राप्त करें।",
        "mr": "तुमच्या शेताच्या ठिकाणासाठी रिअल-टाइम हवामान आणि सोपे पीक मार्गदर्शन मिळवा.",
    },
    "enter_city": {"en": "Enter city", "hi": "शहर दर्ज करें", "mr": "शहर टाका"},
    "check_weather": {"en": "Check Weather", "hi": "मौसम देखें", "mr": "हवामान पहा"},
    "current_weather_summary": {"en": "Current weather summary", "hi": "वर्तमान मौसम सारांश", "mr": "सध्याचा हवामान सारांश"},
    "temperature_label": {"en": "Temperature", "hi": "तापमान", "mr": "तापमान"},
    "farming_insight": {"en": "Farming Insight", "hi": "खेती सुझाव", "mr": "शेती सूचना"},
    "stable_weather_growth": {"en": "Weather conditions look stable for crop growth.", "hi": "फसल वृद्धि के लिए मौसम स्थिर दिख रहा है।", "mr": "पीक वाढीसाठी हवामान स्थिर दिसत आहे."},
    "humidity_fungal_risk": {"en": "High humidity detected. Risk of fungal diseases may increase.", "hi": "उच्च नमी पाई गई। फफूंद रोगों का खतरा बढ़ सकता है।", "mr": "जास्त आर्द्रता आढळली. बुरशीजन्य रोगांचा धोका वाढू शकतो."},
    "low_humidity_irrigation": {"en": "Low humidity detected. Irrigation may be required for crops.", "hi": "कम नमी पाई गई। फसलों को सिंचाई की आवश्यकता हो सकती है।", "mr": "कमी आर्द्रता आढळली. पिकांना पाणी देण्याची गरज लागू शकते."},
    "crops_page_title": {"en": "My Crops", "hi": "मेरी फसलें", "mr": "माझी पिके"},
    "crop_portfolio": {"en": "Crop Portfolio", "hi": "फसल पोर्टफोलियो", "mr": "पीक पोर्टफोलिओ"},
    "crop_portfolio_subtext": {"en": "Track the crops you are currently growing and monitor health insights.", "hi": "आप जो फसलें उगा रहे हैं उन्हें ट्रैक करें और स्वास्थ्य सुझाव देखें।", "mr": "तुम्ही सध्या घेत असलेली पिके ट्रॅक करा आणि आरोग्य सूचना पाहा."},
    "status_healthy": {"en": "Status: Healthy", "hi": "स्थिति: स्वस्थ", "mr": "स्थिती: निरोगी"},
    "status_monitoring": {"en": "Status: Monitoring", "hi": "स्थिति: निगरानी में", "mr": "स्थिती: निरीक्षणात"},
    "status_risk_alert": {"en": "Status: Risk Alert", "hi": "स्थिति: जोखिम अलर्ट", "mr": "स्थिती: धोका इशारा"},
    "insights_page_title": {"en": "AI Insights", "hi": "एआई इनसाइट्स", "mr": "एआय इनसाइट्स"},
    "farming_recommendations": {"en": "Farming Recommendations", "hi": "खेती सिफारिशें", "mr": "शेती शिफारसी"},
    "insights_subtext": {"en": "Key insights generated from crop and environmental conditions.", "hi": "फसल और पर्यावरणीय स्थितियों से बने प्रमुख सुझाव।", "mr": "पीक आणि पर्यावरणीय परिस्थितीतून तयार झालेले महत्त्वाचे सल्ले."},
    "stable_field_moisture": {"en": "Stable field moisture supports healthy crop growth.", "hi": "स्थिर खेत नमी स्वस्थ फसल वृद्धि में मदद करती है।", "mr": "शेतातील स्थिर आर्द्रता निरोगी पीक वाढीस मदत करते."},
    "high_humidity_field": {"en": "High humidity detected in the field environment.", "hi": "खेत के वातावरण में उच्च नमी पाई गई।", "mr": "शेताच्या वातावरणात जास्त आर्द्रता आढळली."},
    "fungal_risk_sensitive": {"en": "Increased risk of fungal diseases for sensitive crops.", "hi": "संवेदनशील फसलों के लिए फफूंद रोगों का जोखिम बढ़ा है।", "mr": "संवेदनशील पिकांसाठी बुरशीजन्य रोगांचा धोका वाढला आहे."},
    "fungicide_tomato": {"en": "Consider preventive fungicide treatment for tomato plants.", "hi": "टमाटर के पौधों के लिए निवारक फफूंदनाशी उपचार पर विचार करें।", "mr": "टोमॅटो पिकासाठी प्रतिबंधात्मक बुरशीनाशक उपचाराचा विचार करा."},
    "login_page_title": {"en": "Farmer Login", "hi": "किसान लॉगिन", "mr": "शेतकरी लॉगिन"},
    "welcome_back": {"en": "Welcome Back", "hi": "फिर से स्वागत है", "mr": "पुन्हा स्वागत आहे"},
    "login_subtext": {"en": "Access your crop dashboard and AI insights.", "hi": "अपना फसल डैशबोर्ड और एआई इनसाइट्स देखें।", "mr": "तुमचा पीक डॅशबोर्ड आणि एआय इनसाइट्स पाहा."},
    "email_address": {"en": "Email Address", "hi": "ईमेल पता", "mr": "ईमेल पत्ता"},
    "password": {"en": "Password", "hi": "पासवर्ड", "mr": "पासवर्ड"},
    "login_button": {"en": "Login", "hi": "लॉगिन", "mr": "लॉगिन"},
    "dont_have_account": {"en": "Don't have an account?", "hi": "क्या आपका खाता नहीं है?", "mr": "तुमचे खाते नाही का?"},
    "create_account_link": {"en": "Create Account", "hi": "खाता बनाएं", "mr": "खाते तयार करा"},
    "signup_page_title": {"en": "Create Account", "hi": "खाता बनाएं", "mr": "खाते तयार करा"},
    "join_farmai": {"en": "Join FarmAI", "hi": "FarmAI से जुड़ें", "mr": "FarmAI मध्ये सामील व्हा"},
    "signup_subtext": {"en": "Create your account to manage crops, weather, and farm planning.", "hi": "फसल, मौसम और योजना प्रबंधन के लिए अपना खाता बनाएं।", "mr": "पीक, हवामान आणि शेती नियोजन व्यवस्थापित करण्यासाठी खाते तयार करा."},
    "full_name": {"en": "Full Name", "hi": "पूरा नाम", "mr": "पूर्ण नाव"},
    "create_account_button": {"en": "Create Account", "hi": "खाता बनाएं", "mr": "खाते तयार करा"},
    "already_have_account": {"en": "Already have an account?", "hi": "क्या आपके पास पहले से खाता है?", "mr": "तुमचे आधीपासून खाते आहे का?"},
    "planner_page_title": {"en": "Planner", "hi": "प्लानर", "mr": "नियोजक"},
    "farmer_assistant": {"en": "Farmer Assistant", "hi": "किसान सहायक", "mr": "शेतकरी सहाय्यक"},
    "planner_header": {"en": "Build your farming calendar with weather-smart planning", "hi": "मौसम आधारित स्मार्ट योजना के साथ अपना खेती कैलेंडर बनाएं", "mr": "हवामान-स्मार्ट नियोजनासह तुमचे शेती कॅलेंडर तयार करा"},
    "planner_header_subtext": {"en": "Add crop activities, review what is due today, and keep an eye on weather risks before field work begins.", "hi": "फसल गतिविधियाँ जोड़ें, आज के काम देखें और खेत में काम शुरू होने से पहले मौसम जोखिम पर नज़र रखें।", "mr": "पीक कामे जोडा, आजची कामे पाहा आणि शेतकाम सुरू करण्यापूर्वी हवामानातील धोक्यांवर लक्ष ठेवा."},
    "risk_alerts_active": {"en": "Risk alerts active", "hi": "जोखिम अलर्ट सक्रिय", "mr": "धोका इशारे सक्रिय"},
    "weather_looks_stable": {"en": "Weather looks stable", "hi": "मौसम स्थिर दिख रहा है", "mr": "हवामान स्थिर दिसते"},
    "add_task": {"en": "Add Task", "hi": "कार्य जोड़ें", "mr": "काम जोडा"},
    "add_task_subtext": {"en": "Save one or more farm activities for a crop and date.", "hi": "एक फसल और तारीख के लिए एक या अधिक खेती कार्य सहेजें।", "mr": "एका पिकासाठी आणि तारखेसाठी एक किंवा अधिक शेती कामे जतन करा."},
    "crop_name": {"en": "Crop name", "hi": "फसल का नाम", "mr": "पिकाचे नाव"},
    "crop_example": {"en": "Example: Tomato", "hi": "उदाहरण: टमाटर", "mr": "उदाहरण: टोमॅटो"},
    "location": {"en": "Location", "hi": "स्थान", "mr": "ठिकाण"},
    "location_example": {"en": "Example: Pune", "hi": "उदाहरण: पुणे", "mr": "उदाहरण: पुणे"},
    "start_date": {"en": "Start date", "hi": "शुरू तारीख", "mr": "सुरुवातीची तारीख"},
    "user_id_optional": {"en": "User ID (optional)", "hi": "यूज़र आईडी (वैकल्पिक)", "mr": "वापरकर्ता आयडी (पर्यायी)"},
    "email_for_updates": {"en": "Email for updates", "hi": "अपडेट के लिए ईमेल", "mr": "अपडेटसाठी ईमेल"},
    "phone_for_sms": {"en": "Phone for SMS alerts", "hi": "एसएमएस अलर्ट के लिए फोन", "mr": "एसएमएस इशाऱ्यांसाठी फोन"},
    "tasks": {"en": "Tasks", "hi": "कार्य", "mr": "कामे"},
    "tasks_placeholder": {"en": "Irrigation\nFertilizer application\nSpraying", "hi": "सिंचाई\nखाद डालना\nछिड़काव", "mr": "पाणी देणे\nखत देणे\nफवारणी"},
    "save_task": {"en": "Save Task", "hi": "कार्य सहेजें", "mr": "काम जतन करा"},
    "weather_title": {"en": "Weather", "hi": "मौसम", "mr": "हवामान"},
    "add_location_to_view_weather": {"en": "Add a location to view weather insights.", "hi": "मौसम जानकारी देखने के लिए स्थान जोड़ें।", "mr": "हवामान माहिती पाहण्यासाठी ठिकाण जोडा."},
    "no_weather_data": {"en": "No weather data available.", "hi": "मौसम डेटा उपलब्ध नहीं है।", "mr": "हवामान डेटा उपलब्ध नाही."},
    "alerts_title": {"en": "Alerts", "hi": "अलर्ट", "mr": "इशारे"},
    "alerts_subtext": {"en": "Rule-based guidance generated from today's weather and tasks.", "hi": "आज के मौसम और कार्यों से बने नियम-आधारित सुझाव।", "mr": "आजच्या हवामान आणि कामांवर आधारित नियमाधारित सूचना."},
    "todays_tasks": {"en": "Today's Tasks", "hi": "आज के कार्य", "mr": "आजची कामे"},
    "scheduled_for": {"en": "Scheduled for {date}", "hi": "{date} के लिए निर्धारित", "mr": "{date} साठी नियोजित"},
    "task_label": {"en": "Task", "hi": "कार्य", "mr": "काम"},
    "date_label": {"en": "Date", "hi": "तारीख", "mr": "तारीख"},
    "location_label": {"en": "Location", "hi": "स्थान", "mr": "ठिकाण"},
    "delete_button": {"en": "Delete", "hi": "हटाएं", "mr": "काढा"},
    "upcoming_tasks": {"en": "Upcoming Tasks", "hi": "आगामी कार्य", "mr": "आगामी कामे"},
    "future_crop_activities": {"en": "Future crop activities sorted by date.", "hi": "तारीख के अनुसार क्रमबद्ध आने वाले कार्य।", "mr": "तारखेनुसार मांडलेली आगामी पीक कामे."},
    "no_tasks_scheduled": {"en": "No tasks scheduled", "hi": "कोई कार्य निर्धारित नहीं", "mr": "कोणतीही कामे नियोजित नाहीत"},
    "invalid_credentials": {"en": "Invalid credentials", "hi": "अमान्य जानकारी", "mr": "अवैध माहिती"},
    "account_created_login": {"en": "Account created. Please login.", "hi": "खाता बन गया। कृपया लॉगिन करें।", "mr": "खाते तयार झाले. कृपया लॉगिन करा."},
    "planner_fields_required": {"en": "Crop, location, date, and at least one task are required.", "hi": "फसल, स्थान, तारीख और कम से कम एक कार्य आवश्यक है।", "mr": "पीक, ठिकाण, तारीख आणि किमान एक काम आवश्यक आहे."},
    "planner_add_task_required": {"en": "Please add at least one farming task.", "hi": "कृपया कम से कम एक खेती कार्य जोड़ें।", "mr": "कृपया किमान एक शेती काम जोडा."},
    "planner_added_tasks": {"en": "Added {count} planner task(s) for {crop}.", "hi": "{crop} के लिए {count} प्लानर कार्य जोड़े गए।", "mr": "{crop} साठी {count} नियोजक कामे जोडली."},
    "planner_task_deleted": {"en": "Task deleted from planner.", "hi": "कार्य प्लानर से हटा दिया गया।", "mr": "काम नियोजकातून काढले गेले."},
    "weather_location_missing": {"en": "Add a planner task with a location to view alerts.", "hi": "अलर्ट देखने के लिए स्थान सहित प्लानर कार्य जोड़ें।", "mr": "इशारे पाहण्यासाठी ठिकाणासह नियोजक काम जोडा."},
    "weather_unavailable_city": {"en": "Weather data is unavailable for {city}. Please verify the city name.", "hi": "{city} के लिए मौसम डेटा उपलब्ध नहीं है। कृपया शहर का नाम जांचें।", "mr": "{city} साठी हवामान डेटा उपलब्ध नाही. कृपया शहराचे नाव तपासा."},
    "high_fungal_alert": {"en": "High fungal disease risk because humidity is above 80%.", "hi": "आर्द्रता 80% से अधिक होने के कारण फफूंद रोग का खतरा अधिक है।", "mr": "आर्द्रता 80% पेक्षा जास्त असल्याने बुरशीजन्य रोगांचा धोका जास्त आहे."},
    "irrigation_recommended_alert": {"en": "Irrigation recommended because temperature is above 35 C.", "hi": "तापमान 35 C से अधिक होने के कारण सिंचाई की सलाह दी जाती है।", "mr": "तापमान 35 C पेक्षा जास्त असल्याने पाणी देण्याची शिफारस केली जाते."},
    "avoid_spraying_alert": {"en": "Avoid pesticide spraying because rain is expected.", "hi": "बारिश की संभावना है, इसलिए कीटनाशक छिड़काव से बचें।", "mr": "पाऊस अपेक्षित असल्याने कीटकनाशक फवारणी टाळा."},
    "tasks_scheduled_today_alert": {"en": "{count} farming task(s) scheduled for today.", "hi": "आज के लिए {count} खेती कार्य निर्धारित हैं।", "mr": "आजसाठी {count} शेती कामे नियोजित आहेत."},
    "no_tasks_today_alert": {"en": "No farming tasks scheduled for today.", "hi": "आज के लिए कोई खेती कार्य निर्धारित नहीं है।", "mr": "आजसाठी कोणतीही शेती कामे नियोजित नाहीत."},
}


def get_locale():
    lang = session.get("lang", "en")
    return lang if lang in LANGUAGES else "en"


def translate_text(key, **kwargs):
    lang = get_locale()
    value = TRANSLATIONS.get(key, {}).get(lang) or TRANSLATIONS.get(key, {}).get("en") or key
    return value.format(**kwargs) if kwargs else value


@app.context_processor
def inject_translation_helpers():
    return {
        "t": translate_text,
        "current_lang": get_locale(),
        "languages": LANGUAGES,
    }


def get_farmer_connection():
    conn = sqlite3.connect(FARMERS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def get_planner_connection():
    conn = sqlite3.connect(PLANNER_DB)
    conn.row_factory = sqlite3.Row
    return conn


def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)
    return img


def add_task(
    crop,
    location,
    task_date,
    task_description,
    user_id=None,
    contact_email=None,
    contact_phone=None,
):
    conn = get_planner_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO tasks(user_id, crop, location, date, task, contact_email, contact_phone)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            crop.strip(),
            location.strip(),
            task_date,
            task_description.strip(),
            (contact_email or "").strip() or None,
            normalize_phone_number(contact_phone),
        ),
    )
    conn.commit()
    conn.close()


def get_tasks_by_date(target_date):
    conn = get_planner_connection()
    rows = conn.execute(
        """
        SELECT id, user_id, crop, location, date, task, contact_email, contact_phone
        FROM tasks
        WHERE date = ?
        ORDER BY crop, task
        """,
        (target_date,),
    ).fetchall()
    conn.close()
    return rows


def get_upcoming_tasks(start_date):
    conn = get_planner_connection()
    rows = conn.execute(
        """
        SELECT id, user_id, crop, location, date, task, contact_email, contact_phone
        FROM tasks
        WHERE date > ?
        ORDER BY date, crop, task
        """,
        (start_date,),
    ).fetchall()
    conn.close()
    return rows


def delete_task(task_id):
    conn = get_planner_connection()
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()


def get_latest_task_location():
    conn = get_planner_connection()
    row = conn.execute(
        """
        SELECT location
        FROM tasks
        ORDER BY date DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    conn.close()
    return row["location"] if row else None


def fetch_weather_snapshot(city):
    if not city:
        return None, translate_text("weather_location_missing")

    try:
        weather_data = get_weather(city)
        return weather_data, None
    except Exception:
        return None, translate_text("weather_unavailable_city", city=city)


def generate_alerts(weather_data, tasks):
    alerts = []
    risky = False

    if weather_data:
        humidity = weather_data.get("humidity", 0)
        temperature = weather_data.get("temperature", 0)
        description = weather_data.get("description", "").lower()

        if humidity > 80:
            alerts.append(
                {
                    "level": "danger",
                    "message": translate_text("high_fungal_alert"),
                }
            )
            risky = True

        if temperature > 35:
            alerts.append(
                {
                    "level": "warning",
                    "message": translate_text("irrigation_recommended_alert"),
                }
            )

        if "rain" in description or "drizzle" in description or "shower" in description:
            alerts.append(
                {
                    "level": "danger",
                    "message": translate_text("avoid_spraying_alert"),
                }
            )
            risky = True

    if tasks:
        alerts.append(
            {
                "level": "info",
                "message": translate_text("tasks_scheduled_today_alert", count=len(tasks)),
            }
        )
    else:
        alerts.append(
            {
                "level": "success",
                "message": translate_text("no_tasks_today_alert"),
            }
        )

    return alerts, risky


def normalize_phone_number(phone_number):
    if not phone_number:
        return None

    cleaned = re.sub(r"[^\d+]", "", phone_number.strip())
    if cleaned.startswith("00"):
        cleaned = f"+{cleaned[2:]}"
    if cleaned.startswith("+"):
        return cleaned
    if cleaned.isdigit():
        return f"+{cleaned}"
    return phone_number.strip()

def send_email(recipient_email, subject, body):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_sender = os.getenv("SMTP_SENDER")

    if not all([smtp_server, smtp_username, smtp_password, smtp_sender]):
        return False, "SMTP settings are not configured."

    if not recipient_email:
        return False, "Recipient email is missing."

    message = MIMEMultipart()
    message["From"] = smtp_sender
    message["To"] = recipient_email
    message["Subject"] = subject

    if isinstance(body, dict):
        plain_body = body.get("plain", "")
        html_body = body.get("html", "")
    else:
        plain_body = body
        html_body = None

    message.attach(MIMEText(plain_body, "plain"))
    if html_body:
        message.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(message)
        return True, "Email sent successfully."
    except Exception as exc:
        return False, f"Email failed: {exc}"


def send_sms(phone_number, message):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN",)
    twilio_number = normalize_phone_number(
        os.getenv("TWILIO_PHONE_NUMBER")
    )
    recipient_number = normalize_phone_number(phone_number)

    if not Client:
        return False, "Twilio package is not installed."

    if not all([account_sid, auth_token, twilio_number]):
        return False, "Twilio settings are not configured."

    if not recipient_number:
        return False, "Recipient phone number is missing."

    if not twilio_number or not twilio_number.startswith("+"):
        return False, "Twilio sender number is invalid."

    if not recipient_number.startswith("+"):
        return False, "Recipient phone number must include country code."

    try:
        client = Client(account_sid, auth_token)
        client.messages.create(body=message, from_=twilio_number, to=recipient_number)
        return True, "SMS sent successfully."
    except Exception as exc:
        return False, f"SMS failed: {exc}"


def build_daily_digest(city, weather_data, tasks, alerts):
    task_lines = [
        f"- {task['date']}: {task['crop']} | {task['task']}" for task in tasks
    ] or ["- No tasks scheduled for today."]
    alert_lines = [f"- {alert['message']}" for alert in alerts] or ["- No alerts."]

    weather_summary = "Weather unavailable."
    if weather_data:
        weather_summary = (
            f"{city}: {weather_data['temperature']} C, "
            f"humidity {weather_data['humidity']}%, "
            f"{weather_data['description']}."
        )

    return (
        "FarmAI Farmer Assistant Daily Summary\n\n"
        f"Weather\n{weather_summary}\n\n"
        "Today's Tasks\n"
        + "\n".join(task_lines)
        + "\n\nWeather Alerts\n"
        + "\n".join(alert_lines)
    )


def build_daily_digest_html(city, weather_data, tasks, alerts):
    if weather_data:
        weather_html = f"""
        <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:16px;">
            <div style="background:#f5fbf6;border:1px solid #dbe8dd;border-radius:14px;padding:14px;">
                <div style="font-size:12px;color:#6c7f72;text-transform:uppercase;letter-spacing:.08em;">Temperature</div>
                <div style="font-size:24px;font-weight:700;color:#184f32;margin-top:6px;">{weather_data['temperature']} C</div>
            </div>
            <div style="background:#f5fbf6;border:1px solid #dbe8dd;border-radius:14px;padding:14px;">
                <div style="font-size:12px;color:#6c7f72;text-transform:uppercase;letter-spacing:.08em;">Humidity</div>
                <div style="font-size:24px;font-weight:700;color:#184f32;margin-top:6px;">{weather_data['humidity']}%</div>
            </div>
            <div style="background:#f5fbf6;border:1px solid #dbe8dd;border-radius:14px;padding:14px;">
                <div style="font-size:12px;color:#6c7f72;text-transform:uppercase;letter-spacing:.08em;">Condition</div>
                <div style="font-size:18px;font-weight:600;color:#184f32;margin-top:6px;text-transform:capitalize;">{weather_data['description']}</div>
            </div>
        </div>
        """
    else:
        weather_html = """
        <div style="margin-top:16px;padding:14px 16px;border-radius:14px;background:#f8faf8;border:1px dashed #cbd9cf;color:#607364;">
            Weather data is unavailable right now.
        </div>
        """

    task_items = "".join(
        f"""
        <div style="padding:14px 16px;border-radius:14px;background:#f8fcf8;border:1px solid #deebdf;border-left:5px solid #2c7a4b;margin-top:10px;">
            <div style="font-size:18px;font-weight:700;color:#1b5638;">{task['crop']}</div>
            <div style="margin-top:6px;color:#4f6554;"><strong>Task:</strong> {task['task']}</div>
            <div style="margin-top:4px;color:#4f6554;"><strong>Date:</strong> {task['date']}</div>
            <div style="margin-top:4px;color:#4f6554;"><strong>Location:</strong> {task['location']}</div>
        </div>
        """
        for task in tasks
    ) or """
    <div style="margin-top:16px;padding:14px 16px;border-radius:14px;background:#f8faf8;border:1px dashed #cbd9cf;color:#607364;">
        No tasks scheduled for today.
    </div>
    """

    def alert_style(level):
        if level == "danger":
            return "#fff1ee", "#ffd8cf", "#a33e2f"
        if level == "warning":
            return "#fff8e9", "#f5dd9c", "#8a6215"
        if level == "success":
            return "#eef9ef", "#d1ecd5", "#216640"
        return "#edf6ff", "#d1e5fb", "#2d67af"

    alert_items = "".join(
        f"""
        <div style="margin-top:10px;padding:14px 16px;border-radius:14px;background:{bg};border:1px solid {border};color:{text};">
            <strong style="display:inline-block;margin-right:8px;text-transform:uppercase;font-size:12px;letter-spacing:.08em;">{alert['level']}</strong>
            {alert['message']}
        </div>
        """
        for alert in alerts
        for bg, border, text in [alert_style(alert["level"])]
    )

    return f"""
    <html>
    <body style="margin:0;padding:24px;background:#edf6ef;font-family:Poppins,Arial,sans-serif;color:#1d3124;">
        <div style="max-width:720px;margin:0 auto;">
            <div style="background:linear-gradient(135deg,#1f5a3a 0%,#327b52 55%,#8bcf96 100%);border-radius:24px;padding:28px 30px;color:#ffffff;box-shadow:0 18px 40px rgba(35,91,60,.18);">
                <div style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;background:rgba(255,255,255,.14);display:inline-block;padding:6px 12px;border-radius:999px;">FarmAI Daily Alert</div>
                <h1 style="margin:14px 0 8px;font-size:30px;line-height:1.15;">Farmer Assistant Summary</h1>
                <p style="margin:0;color:rgba(255,255,255,.9);font-size:15px;">Your daily farming plan and weather-based alerts for <strong>{city}</strong>.</p>
            </div>

            <div style="background:#ffffff;border-radius:20px;padding:24px;margin-top:18px;box-shadow:0 12px 30px rgba(33,75,50,.08);">
                <h2 style="margin:0;font-size:22px;color:#184f32;">Weather Overview</h2>
                {weather_html}
            </div>

            <div style="background:#ffffff;border-radius:20px;padding:24px;margin-top:18px;box-shadow:0 12px 30px rgba(33,75,50,.08);">
                <h2 style="margin:0;font-size:22px;color:#184f32;">Today's Tasks</h2>
                {task_items}
            </div>

            <div style="background:#ffffff;border-radius:20px;padding:24px;margin-top:18px;box-shadow:0 12px 30px rgba(33,75,50,.08);">
                <h2 style="margin:0;font-size:22px;color:#184f32;">Weather Alerts</h2>
                {alert_items}
            </div>
        </div>
    </body>
    </html>
    """


def run_daily_planner_notifications():
    today = date.today().isoformat()
    today_tasks = get_tasks_by_date(today)

    if not today_tasks:
        return

    grouped_contacts = defaultdict(list)
    for task in today_tasks:
        key = (
            task["location"],
            task["contact_email"],
            task["contact_phone"],
        )
        grouped_contacts[key].append(task)

    for (location, contact_email, contact_phone), tasks in grouped_contacts.items():
        weather_data, _ = fetch_weather_snapshot(location)
        alerts, risky = generate_alerts(weather_data, tasks)
        message = {
            "plain": build_daily_digest(location, weather_data, tasks, alerts),
            "html": build_daily_digest_html(location, weather_data, tasks, alerts),
        }

        if contact_email:
            send_email(contact_email, "FarmAI Daily Planner Alert", message)

        if risky and contact_phone:
            sms_lines = [alert["message"] for alert in alerts if alert["level"] == "danger"]
            sms_sent, sms_status = send_sms(
                contact_phone,
                f"FarmAI alert for {location}: {' '.join(sms_lines)}",
            )
            if not sms_sent:
                print(sms_status)


def configure_scheduler():
    if not scheduler.running:
        scheduler.add_job(
            run_daily_planner_notifications,
            trigger="cron",
            hour=7,
            minute=0,
            id="planner_daily_notifications",
            replace_existing=True,
        )
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown(wait=False))


def build_planner_context(selected_city=None):
    today = date.today().isoformat()
    today_tasks = get_tasks_by_date(today)
    upcoming_tasks = get_upcoming_tasks(today)
    location = selected_city or get_latest_task_location()
    weather_data, weather_error = fetch_weather_snapshot(location)
    alerts, risky = generate_alerts(weather_data, today_tasks)

    return {
        "selected_city": location,
        "weather_data": weather_data,
        "weather_error": weather_error,
        "today_tasks": today_tasks,
        "upcoming_tasks": upcoming_tasks,
        "alerts": alerts,
        "has_risky_alert": risky,
        "today_date": today,
    }


@app.route("/set-language", methods=["POST"])
def set_language():
    selected_lang = request.form.get("language", "en")
    session["lang"] = selected_lang if selected_lang in LANGUAGES else "en"
    return redirect(request.form.get("next") or request.referrer or url_for("home"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/scan")
def scan():
    return render_template("scan_crop.html")


@app.route("/crops")
def crops():
    return render_template("crops.html")


@app.route("/insights")
def insights():
    return render_template("insights.html")


@app.route("/planner", methods=["GET", "POST"])
def planner():
    if request.method == "POST":
        crop = request.form.get("crop", "").strip()
        location = request.form.get("location", "").strip()
        start_date = request.form.get("start_date", "").strip()
        task_text = request.form.get("task_description", "").strip()
        user_id = request.form.get("user_id", "").strip() or None
        contact_email = request.form.get("contact_email", "").strip()
        contact_phone = request.form.get("contact_phone", "").strip()

        if not all([crop, location, start_date, task_text]):
            flash(translate_text("planner_fields_required"), "danger")
            return redirect(url_for("planner"))

        task_lines = [line.strip() for line in task_text.splitlines() if line.strip()]
        if not task_lines:
            flash(translate_text("planner_add_task_required"), "danger")
            return redirect(url_for("planner"))

        for task_description in task_lines:
            add_task(
                crop=crop,
                location=location,
                task_date=start_date,
                task_description=task_description,
                user_id=int(user_id) if user_id else None,
                contact_email=contact_email,
                contact_phone=contact_phone,
            )

        flash(translate_text("planner_added_tasks", count=len(task_lines), crop=crop), "success")
        return redirect(url_for("planner", city=location))

    context = build_planner_context(request.args.get("city"))
    return render_template("planner.html", **context)


@app.route("/planner/delete/<int:task_id>", methods=["POST"])
def delete_planner_task(task_id):
    selected_city = request.form.get("city") or request.args.get("city")
    delete_task(task_id)
    flash(translate_text("planner_task_deleted"), "success")
    if selected_city:
        return redirect(url_for("planner", city=selected_city))
    return redirect(url_for("planner"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_farmer_connection()
        user = conn.execute(
            "SELECT * FROM farmers WHERE email=? AND password=?",
            (email, password),
        ).fetchone()
        conn.close()

        if user:
            return render_template("index.html", user=user["name"])

        return render_template("login.html", error=translate_text("invalid_credentials"))

    return render_template("login.html")


@app.route("/weather", methods=["GET", "POST"])
def weather():
    if request.method == "POST":
        city = request.form["city"]
        data, error = fetch_weather_snapshot(city)

        if error:
            return render_template("weather.html", error=error, city=city)

        return render_template(
            "weather.html",
            temp=data["temperature"],
            feels=data["feels_like"],
            humidity=data["humidity"],
            wind=data["wind"],
            description=data["description"],
            icon=data["icon"],
            city=city,
        )

    return render_template("weather.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        conn = get_farmer_connection()
        conn.execute(
            "INSERT INTO farmers(name,email,password) VALUES(?,?,?)",
            (name, email, password),
        )
        conn.commit()
        conn.close()

        return render_template("login.html", message=translate_text("account_created_login"))

    return render_template("signup.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    processed_img = preprocess(img)
    prediction = model.predict(processed_img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    clean_name = predicted_class.replace("___", " ").replace("_", " ")

    if "healthy" in predicted_class.lower():
        cause = "No disease detected."
        treatment = "Plant is healthy. No treatment required."
        pesticides = []
    else:
        info = advisory.get(predicted_class, {})
        cause = info.get("cause", "Information not available")
        treatment = info.get("treatment", "Consult agriculture expert")
        pesticides = info.get("pesticides", [])

    return render_template(
        "scan_crop.html",
        prediction=clean_name,
        confidence=round(confidence, 2),
        cause=cause,
        treatment=treatment,
        pesticides=pesticides,
    )


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        configure_scheduler()
    app.run(debug=True)
