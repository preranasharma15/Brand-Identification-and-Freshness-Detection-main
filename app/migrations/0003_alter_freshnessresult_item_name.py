# Generated by Django 5.1.4 on 2024-12-16 09:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_remove_brandresult_brand_name_remove_brandresult_id_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='freshnessresult',
            name='item_name',
            field=models.CharField(default='Unknown', primary_key=True, serialize=False),
        ),
    ]
