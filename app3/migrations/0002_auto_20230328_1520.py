# Generated by Django 3.1.6 on 2023-03-28 19:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app3', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='posttrainingmodel3d',
            old_name='trans0',
            new_name='trans6',
        ),
        migrations.RenameField(
            model_name='posttrainingmodel3d',
            old_name='trust0',
            new_name='trans7',
        ),
        migrations.RenameField(
            model_name='questionnairemodel3d',
            old_name='trans0',
            new_name='trans6',
        ),
        migrations.RenameField(
            model_name='questionnairemodel3d',
            old_name='trust0',
            new_name='trans7',
        ),
        migrations.AddField(
            model_name='posttrainingmodel3d',
            name='trust5',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='posttrainingmodel3d',
            name='trust6',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='posttrainingmodel3d',
            name='trust7',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='questionnairemodel3d',
            name='trust5',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='questionnairemodel3d',
            name='trust6',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='questionnairemodel3d',
            name='trust7',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
