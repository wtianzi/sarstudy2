# Generated by Django 3.1.6 on 2023-03-13 19:41

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ClueMedia',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(blank=True, max_length=255, null=True)),
                ('longitude', models.DecimalField(blank=True, decimal_places=9, max_digits=12, null=True)),
                ('latitude', models.DecimalField(blank=True, decimal_places=9, max_digits=12, null=True)),
                ('photo', models.ImageField(blank=True, default='No-img.png', null=True, upload_to='uploads/')),
                ('description', models.CharField(blank=True, max_length=100, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='DataStorage',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('taskid', models.CharField(blank=True, max_length=100, null=True)),
                ('subtaskid', models.CharField(blank=True, max_length=100, null=True)),
                ('data', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='DemographicsModel',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('participantid', models.CharField(blank=True, max_length=100, null=True)),
                ('participantindex', models.IntegerField(blank=True, null=True)),
                ('age', models.IntegerField(blank=True, null=True)),
                ('gender', models.IntegerField(blank=True, null=True)),
                ('education', models.IntegerField(blank=True, null=True)),
                ('sart', models.IntegerField(blank=True, null=True)),
                ('q1', models.IntegerField(blank=True, null=True)),
                ('q2', models.CharField(blank=True, max_length=100, null=True)),
                ('q3', models.CharField(blank=True, max_length=100, null=True)),
                ('q4', models.TextField(blank=True, null=True)),
                ('q5', models.TextField(blank=True, null=True)),
                ('q6', models.TextField(blank=True, null=True)),
                ('q7', models.TextField(blank=True, null=True)),
                ('q8', models.TextField(blank=True, null=True)),
                ('q9', models.TextField(blank=True, null=True)),
                ('q10', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='DemographicsModel3D',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('participantid', models.CharField(blank=True, max_length=100, null=True)),
                ('participantindex', models.IntegerField(blank=True, null=True)),
                ('age', models.IntegerField(blank=True, null=True)),
                ('gender', models.IntegerField(blank=True, null=True)),
                ('education', models.IntegerField(blank=True, null=True)),
                ('sart', models.IntegerField(blank=True, null=True)),
                ('q1', models.IntegerField(blank=True, null=True)),
                ('q2', models.CharField(blank=True, max_length=100, null=True)),
                ('q3', models.CharField(blank=True, max_length=100, null=True)),
                ('q4', models.TextField(blank=True, null=True)),
                ('q5', models.TextField(blank=True, null=True)),
                ('q6', models.TextField(blank=True, null=True)),
                ('q7', models.TextField(blank=True, null=True)),
                ('q8', models.TextField(blank=True, null=True)),
                ('q9', models.TextField(blank=True, null=True)),
                ('q10', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='ExperimentDataStorage',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('details', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='GPSData',
            fields=[
                ('deviceid', models.CharField(max_length=20, primary_key=True, serialize=False)),
                ('taskid', models.CharField(blank=True, max_length=100, null=True)),
                ('gpsdata', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='GPShistoricalData',
            fields=[
                ('deviceid', models.CharField(max_length=20, primary_key=True, serialize=False)),
                ('taskid', models.CharField(blank=True, max_length=100, null=True)),
                ('gpshistoricaldata', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='ParticipantStatusModel',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('participantid', models.CharField(blank=True, max_length=100, null=True)),
                ('participantindex', models.IntegerField(blank=True, null=True)),
                ('participantname', models.CharField(blank=True, max_length=100, null=True)),
                ('status', models.BooleanField(default=False)),
                ('taskstatus', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Person',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('first_name', models.CharField(max_length=30)),
                ('last_name', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='PostExpSurveyModel',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('participantid', models.CharField(blank=True, max_length=100, null=True)),
                ('q1', models.TextField(blank=True, null=True)),
                ('q2', models.TextField(blank=True, null=True)),
                ('q3', models.TextField(blank=True, null=True)),
                ('q4', models.TextField(blank=True, null=True)),
                ('q5', models.TextField(blank=True, null=True)),
                ('q6', models.TextField(blank=True, null=True)),
                ('q7', models.TextField(blank=True, null=True)),
                ('q8', models.TextField(blank=True, null=True)),
                ('q9', models.TextField(blank=True, null=True)),
                ('q10', models.TextField(blank=True, null=True)),
                ('q11', models.TextField(blank=True, null=True)),
                ('q12', models.TextField(blank=True, null=True)),
                ('q13', models.TextField(blank=True, null=True)),
                ('q14', models.TextField(blank=True, null=True)),
                ('q15', models.TextField(blank=True, null=True)),
                ('q16', models.TextField(blank=True, null=True)),
                ('q17', models.TextField(blank=True, null=True)),
                ('q18', models.TextField(blank=True, null=True)),
                ('q19', models.TextField(blank=True, null=True)),
                ('q20', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='PostExpSurveyModel3D',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('participantid', models.CharField(blank=True, max_length=100, null=True)),
                ('q1', models.TextField(blank=True, null=True)),
                ('q2', models.TextField(blank=True, null=True)),
                ('q3', models.TextField(blank=True, null=True)),
                ('q4', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='PostTrainingModel3D',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('participantid', models.CharField(blank=True, max_length=100, null=True)),
                ('taskid', models.CharField(blank=True, max_length=100, null=True)),
                ('sceneid', models.CharField(blank=True, max_length=100, null=True)),
                ('trust0', models.IntegerField(blank=True, null=True)),
                ('trust1', models.IntegerField(blank=True, null=True)),
                ('trust2', models.IntegerField(blank=True, null=True)),
                ('trust3', models.IntegerField(blank=True, null=True)),
                ('trust4', models.IntegerField(blank=True, null=True)),
                ('trans0', models.IntegerField(blank=True, null=True)),
                ('trans1', models.IntegerField(blank=True, null=True)),
                ('trans2', models.IntegerField(blank=True, null=True)),
                ('trans3', models.IntegerField(blank=True, null=True)),
                ('trans4', models.IntegerField(blank=True, null=True)),
                ('trans5', models.IntegerField(blank=True, null=True)),
                ('workload', models.IntegerField(blank=True, null=True)),
                ('NASATLX1_mental', models.IntegerField(blank=True, null=True)),
                ('NASATLX2_physical', models.IntegerField(blank=True, null=True)),
                ('NASATLX3_temporal', models.IntegerField(blank=True, null=True)),
                ('NASATLX4_performance', models.IntegerField(blank=True, null=True)),
                ('NASATLX5_effort', models.IntegerField(blank=True, null=True)),
                ('NASATLX6_frustration', models.IntegerField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='QuestionnaireModel',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('participantid', models.CharField(blank=True, max_length=100, null=True)),
                ('taskid', models.CharField(blank=True, max_length=100, null=True)),
                ('sceneid', models.CharField(blank=True, max_length=100, null=True)),
                ('trust', models.IntegerField(blank=True, null=True)),
                ('transparency', models.IntegerField(blank=True, null=True)),
                ('workload', models.IntegerField(blank=True, null=True)),
                ('trans1', models.IntegerField(blank=True, null=True)),
                ('trans2', models.IntegerField(blank=True, null=True)),
                ('trans3', models.IntegerField(blank=True, null=True)),
                ('trans4', models.IntegerField(blank=True, null=True)),
                ('trans5', models.IntegerField(blank=True, null=True)),
                ('trust1', models.IntegerField(blank=True, null=True)),
                ('trust2', models.IntegerField(blank=True, null=True)),
                ('trust3', models.IntegerField(blank=True, null=True)),
                ('trust4', models.IntegerField(blank=True, null=True)),
                ('trust5', models.IntegerField(blank=True, null=True)),
                ('NASATLX1_mental', models.IntegerField(blank=True, null=True)),
                ('NASATLX2_physical', models.IntegerField(blank=True, null=True)),
                ('NASATLX3_temporal', models.IntegerField(blank=True, null=True)),
                ('NASATLX4_performance', models.IntegerField(blank=True, null=True)),
                ('NASATLX5_effort', models.IntegerField(blank=True, null=True)),
                ('NASATLX6_frustration', models.IntegerField(blank=True, null=True)),
                ('q1', models.IntegerField(blank=True, null=True)),
                ('q2', models.IntegerField(blank=True, null=True)),
                ('q3', models.IntegerField(blank=True, null=True)),
                ('q4', models.IntegerField(blank=True, null=True)),
                ('q5', models.IntegerField(blank=True, null=True)),
                ('q6', models.IntegerField(blank=True, null=True)),
                ('q7', models.IntegerField(blank=True, null=True)),
                ('q8', models.IntegerField(blank=True, null=True)),
                ('q9', models.IntegerField(blank=True, null=True)),
                ('q10', models.IntegerField(blank=True, null=True)),
                ('q11', models.IntegerField(blank=True, null=True)),
                ('q12', models.IntegerField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='QuestionnaireModel3D',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('participantid', models.CharField(blank=True, max_length=100, null=True)),
                ('taskid', models.CharField(blank=True, max_length=100, null=True)),
                ('sceneid', models.CharField(blank=True, max_length=100, null=True)),
                ('trust0', models.IntegerField(blank=True, null=True)),
                ('trust1', models.IntegerField(blank=True, null=True)),
                ('trust2', models.IntegerField(blank=True, null=True)),
                ('trust3', models.IntegerField(blank=True, null=True)),
                ('trust4', models.IntegerField(blank=True, null=True)),
                ('trans0', models.IntegerField(blank=True, null=True)),
                ('trans1', models.IntegerField(blank=True, null=True)),
                ('trans2', models.IntegerField(blank=True, null=True)),
                ('trans3', models.IntegerField(blank=True, null=True)),
                ('trans4', models.IntegerField(blank=True, null=True)),
                ('trans5', models.IntegerField(blank=True, null=True)),
                ('workload', models.IntegerField(blank=True, null=True)),
                ('NASATLX1_mental', models.IntegerField(blank=True, null=True)),
                ('NASATLX2_physical', models.IntegerField(blank=True, null=True)),
                ('NASATLX3_temporal', models.IntegerField(blank=True, null=True)),
                ('NASATLX4_performance', models.IntegerField(blank=True, null=True)),
                ('NASATLX5_effort', models.IntegerField(blank=True, null=True)),
                ('NASATLX6_frustration', models.IntegerField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Task',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('taskpolygon', models.TextField(blank=True, null=True)),
                ('notes', models.CharField(max_length=30)),
                ('taskid', models.CharField(blank=True, max_length=100, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='TaskAssignment',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('resourcetype', models.CharField(blank=True, max_length=100, null=True)),
                ('planningno', models.CharField(blank=True, max_length=100, null=True)),
                ('priority', models.CharField(blank=True, max_length=100, null=True)),
                ('task_complete', models.BooleanField(default=True)),
                ('task_partially_finished', models.BooleanField(default=True)),
                ('urgent_follow_up', models.BooleanField(default=True)),
                ('task_number', models.CharField(blank=True, default='0000', max_length=100)),
                ('team_identifier', models.CharField(blank=True, max_length=100, null=True)),
                ('resource_type', models.CharField(blank=True, max_length=100, null=True)),
                ('task_map', models.CharField(blank=True, max_length=100, null=True)),
                ('branch', models.CharField(blank=True, max_length=100, null=True)),
                ('division_group', models.CharField(blank=True, max_length=100, null=True)),
                ('incident_name', models.CharField(blank=True, max_length=100, null=True)),
                ('task_instructions', models.TextField(blank=True, null=True)),
                ('previous_search', models.CharField(blank=True, max_length=1000, null=True)),
                ('transportation', models.CharField(blank=True, max_length=1000, null=True)),
                ('equipment_requirements', models.CharField(blank=True, max_length=1000, null=True)),
                ('expected_time_frame', models.BooleanField(default=True)),
                ('expected_time_frame_input', models.CharField(blank=True, max_length=100, null=True)),
                ('target_pod_subject', models.BooleanField(default=True)),
                ('target_pod_subject_input', models.CharField(blank=True, max_length=100, null=True)),
                ('target_pod_clues', models.BooleanField(default=True)),
                ('target_pod_clues_input', models.CharField(blank=True, max_length=100, null=True)),
                ('team_nearby', models.BooleanField(default=True)),
                ('team_nearby_input', models.CharField(blank=True, max_length=100, null=True)),
                ('applicable_clues', models.BooleanField(default=True)),
                ('terrain_hazrds', models.BooleanField(default=True)),
                ('weather_safety_issues', models.BooleanField(default=True)),
                ('press_family_plans', models.BooleanField(default=True)),
                ('subject_information', models.BooleanField(default=True)),
                ('rescue_find_plans', models.BooleanField(default=True)),
                ('others', models.BooleanField(default=True)),
                ('others_input', models.TextField(blank=True, null=True)),
                ('team_call_sign', models.CharField(blank=True, max_length=100, null=True)),
                ('freq_team', models.CharField(blank=True, max_length=100, null=True)),
                ('base_call_sign', models.CharField(blank=True, max_length=100, null=True)),
                ('freq_base', models.CharField(blank=True, max_length=100, null=True)),
                ('pertinent_phone_no', models.CharField(blank=True, max_length=100, null=True)),
                ('base', models.CharField(blank=True, max_length=100, null=True)),
                ('check_in_feq', models.CharField(blank=True, max_length=100, null=True)),
                ('check_in_hour', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_1_function', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_1_freq', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_1_comments', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_2_function', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_2_freq', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_2_comments', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_3_function', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_3_freq', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_3_comments', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_4_function', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_4_freq', models.CharField(blank=True, max_length=100, null=True)),
                ('tactical_4_comments', models.CharField(blank=True, max_length=100, null=True)),
                ('note_safety_message', models.TextField(blank=True, null=True)),
                ('prepared_by', models.CharField(blank=True, max_length=100, null=True)),
                ('briefed_by', models.CharField(blank=True, max_length=100, null=True)),
                ('time_out', models.CharField(blank=True, max_length=100, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='WaypointsData',
            fields=[
                ('deviceid', models.CharField(max_length=20, primary_key=True, serialize=False)),
                ('taskid', models.CharField(blank=True, max_length=100, null=True)),
                ('waypointsdata', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='WebapplicationModel',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('participantid', models.CharField(blank=True, max_length=100, null=True)),
                ('q1', models.IntegerField(blank=True, null=True)),
                ('q2', models.IntegerField(blank=True, null=True)),
                ('q3', models.IntegerField(blank=True, null=True)),
                ('q4', models.IntegerField(blank=True, null=True)),
                ('q5', models.IntegerField(blank=True, null=True)),
                ('q6', models.IntegerField(blank=True, null=True)),
                ('q7', models.IntegerField(blank=True, null=True)),
                ('q8', models.IntegerField(blank=True, null=True)),
                ('q9', models.IntegerField(blank=True, null=True)),
                ('q10', models.IntegerField(blank=True, null=True)),
                ('q11', models.IntegerField(blank=True, null=True)),
                ('q12', models.IntegerField(blank=True, null=True)),
                ('q13', models.IntegerField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True, null=True)),
            ],
        ),
    ]
