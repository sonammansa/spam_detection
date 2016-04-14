from django.conf.urls import patterns, url

urlpatterns = patterns('hotels.views',
                       url(r'^/$', 'index', name='spam_detection'),
		      )
