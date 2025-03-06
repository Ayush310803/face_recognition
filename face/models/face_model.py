from mongoengine import Document, StringField, DateTimeField
from datetime import datetime

class FaceImage(Document):
    name = StringField(required=True)
    image_path = StringField()
    timestamp = DateTimeField(default=datetime.utcnow)