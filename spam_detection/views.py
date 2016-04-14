from django.shortcuts import render
from django.shortcuts import render_to_response
from django.template import RequestContext
# Create your views here.

def index(request):
	context = {'data': ''}
	return render_to_response('index.html', context, context_instance=RequestContext(request))