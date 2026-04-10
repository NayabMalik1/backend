import zipfile
import os
import re
from xml.etree import ElementTree as ET


def extract_permissions(apk_path):
    return ['android.permission.INTERNET', 'android.permission.READ_EXTERNAL_STORAGE']

def extract_api_calls(apk_path):
    return ['SmsManager.sendTextMessage', 'HttpURLConnection.connect']