# Generated by Django 4.1.7 on 2023-04-12 19:52

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app3', '0003_auto_20230331_2029'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='demographicsmodel',
            name='updated_at',
        ),
        migrations.RemoveField(
            model_name='demographicsmodel3d',
            name='updated_at',
        ),
        migrations.RemoveField(
            model_name='participantstatusmodel',
            name='updated_at',
        ),
        migrations.RemoveField(
            model_name='postexpsurveymodel',
            name='updated_at',
        ),
        migrations.RemoveField(
            model_name='postexpsurveymodel3d',
            name='updated_at',
        ),
        migrations.RemoveField(
            model_name='posttrainingmodel3d',
            name='updated_at',
        ),
        migrations.RemoveField(
            model_name='questionnairemodel',
            name='updated_at',
        ),
        migrations.RemoveField(
            model_name='questionnairemodel3d',
            name='updated_at',
        ),
        migrations.RemoveField(
            model_name='taskassignment',
            name='updated_at',
        ),
        migrations.RemoveField(
            model_name='webapplicationmodel',
            name='updated_at',
        ),
    ]