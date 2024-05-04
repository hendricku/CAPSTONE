from django.urls import path 
from camera_to_text import views 
urlpatterns = [
	path("", views.index, name="index"),
    path("real_time", views.real_time, name="real_time"),
	path('perform_fruit_detection/', views.perform_fruit_detection, name='perform_fruit_detection'), #### bagong url use for prediction
    path('save_image/', views.save_image, name='save_image'),
    # path('perform_text_to_speech/', views.perform_text_to_speech, name='perform_text_to_speech'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('capture_image/', views.capture_image, name='capture_image'), 
    path('classify/', views.classify_image, name='classify_image'),  ### url connected sa r_detection.html video for real time detection 
	path('speech_to_text/', views.speech_to_text, name='speech_to_text'),
    ]

