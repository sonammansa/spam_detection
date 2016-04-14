from django.conf.urls import patterns, url

urlpatterns = patterns('spam_detection.views',
                       url(r'^$', 'index', name='spam_detection'),
		      )
