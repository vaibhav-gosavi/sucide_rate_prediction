from django.db import models

class Prediction(models.Model):
    model_type = models.CharField(max_length=50)
    prediction = models.FloatField()
    actual_value = models.FloatField()
    mae = models.FloatField()
    mse = models.FloatField()
    rmse = models.FloatField()
    r2 = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
