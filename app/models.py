from django.db import models

class BrandResult(models.Model):
    timestamp = models.DateTimeField(primary_key=True, auto_now_add=True)
    brand = models.CharField(max_length=255, default="Unknown")
    

class FreshnessResult(models.Model):
    timestamp = models.DateTimeField(primary_key=True, auto_now_add=True)
    item_name = models.CharField(default="Unknown")
    freshness_score = models.FloatField(default=0.0)
    shelf_life =  models.IntegerField(default=1)
